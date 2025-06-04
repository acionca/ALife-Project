import numpy as np
import torch
from pyca import Automaton
from matplotlib.colors import hsv_to_rgb, to_rgb
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances

import matplotlib.pyplot as plt


class PlanetaryCA(Automaton):
    def __init__(
        self,
        size,
        n_states: int = 2,
        n_active: int = 4,
        dt: float = 1,
        d_thresh: float = 1,
        r0: float = 2,
        seed: float = None,
        scale: float = 40,
        ca_time: int = 100,
        bouncing: bool = False,
        growth: float = 1.0,
        device: str = "cpu",
    ):
        super().__init__(size)

        self.device = device

        if seed is not None:
            torch.manual_seed(seed)

        # Plotting parameters
        self.el_size = 200
        # self.scale = scale
        self.radius = 4

        self.box_size = torch.tensor([2 * self.h, 2 * self.w], dtype=torch.float32)
        self.scale = self.box_size

        # CA Parameters
        self.n_states = n_states
        self.n_active = n_active
        self.state_t = torch.zeros(self.h * self.w, dtype=torch.float32)
        self.mass_max = 10

        # Kinetics
        self.d_thresh = d_thresh
        self.r0 = r0
        self.k_attract = 1.0
        self.k_repel = 4.0
        self.ca_time = ca_time
        self.dt = dt
        self.bouncing = bouncing
        self.growth = torch.tensor(growth)
        self._new_kinetics()

        self._new_world()
        self.m[self.world.sum(dim=1) > 0] = 1

        self._gen_colors()

    def _new_kinetics(self):
        """
        Generates initial kinetics (initialization).
        """
        self.positions = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(self.h), torch.arange(self.w), indexing="ij"
                ),
                dim=-1,
            )
            .to(self.device)
            .float()
        )  # (H,W,2), positions[x,y] = (x,y)
        self.positions += torch.tensor([self.h / 2 + 0.5, self.w / 2 + 0.5])
        self.pos_vec = self.positions.view((-1, 2)).float()

        self.v = torch.zeros(
            (self.h * self.w, 2), dtype=torch.float32
        )  # (H * W, 2), velocity of each cell
        # self.m = 0.5*torch.ones(self.h * self.w, dtype=torch.float32) # (H * W), mass of each cell
        self.m = torch.ones(
            self.h * self.w, dtype=torch.float32
        )  # (H * W), mass of each cell
        self.time = 0

    def _new_world(self):
        """
        Generates a new world (initialization).
        """
        self.active_cells = torch.multinomial(
            torch.ones(self.h * self.w), self.n_active, replacement=False
        )
        self.world = torch.zeros(
            (self.h * self.w, self.n_states), dtype=torch.int, device=self.device
        )

        for i in range(self.n_states):
            self.world[self.active_cells[i :: self.n_states], i] = 1

        self.graph = nx.grid_2d_graph(
            self.h, self.w
        )  # Create a grid graph for the automaton

        self._new_kinetics()
        self._compute_forces()

        self.dirac = torch.zeros(self.n_states)
        self.dirac[0] = 1

    def _gen_colors(self, cmap="coolwarm"):
        """
        Generates a colormap for the automaton.
        """

        cmap = plt.get_cmap("coolwarm", 3)
        white_col = torch.tensor(to_rgb(cmap(1)[:3]))

        if self.n_states == 2:
            cmap = plt.get_cmap(cmap, self.n_states + 1)
            coldic = {
                0: white_col,
                1: torch.tensor(to_rgb(cmap(0)[:3])),
                2: torch.tensor(to_rgb(cmap(2)[:3])),
            }
        elif self.n_states > 2:
            cmap = plt.get_cmap("turbo", self.n_states + 1)
            coldic = {0: white_col}
            coldic.update(
                {
                    i: torch.tensor(to_rgb(cmap(0)[:3]))
                    for i in range(1, self.n_states + 1)
                }
            )
        else:
            return NotImplementedError(
                "Colors not implemented for n_states other than 2"
            )
        self.color_dict = coldic

    def _get_neighbors(self):
        neighbor_mask = (self.dist < self.d_thresh) & (self.dist > 0)  # exclude self

        neighbors = [
            np.where(neighbor_mask[i])[0].tolist() for i in range(len(self.pos_vec))
        ]
        return neighbors

    def _get_force_signs(self):
        state = (self.world * torch.tensor([1, 2])).sum(dim=1)

        # Outer comparison matrix
        Si = state[:, None]
        Sj = state[None, :]

        a_mask = (Si == Sj) * (Si * Sj > 0)
        a_mask.fill_diagonal_(False)  # Exclude self-comparison

        r_mask = Si != Sj
        r_mask.fill_diagonal_(False)  # Exclude self-comparison

        return a_mask, r_mask

    def _compute_forces(self, softening=1e-2):
        delta = (
            self.pos_vec[:, np.newaxis, :] - self.pos_vec[np.newaxis, :, :]
        )  # (N, N, 2)

        # Minimum image convention for periodic boundaries
        if not self.bouncing:
            delta = (delta + self.box_size / 2) % self.box_size - self.box_size / 2

        # self.dist = np.linalg.norm(delta, axis=2) + softening
        self.dist = torch.norm(delta, dim=2) + softening  # (N, N)
        self.dist.fill_diagonal_(1.0)

        # TODO: Use f = -6 * tau^6 * (tau^6 / r^6  - 1) / r^7

        # TODO: r^2 instead of r for accurate force calculation
        # inv_dist = 1.0 / self.dist
        # inv_dist = 1.0 / self.dist.pow(2)
        m_product = self.m[:, np.newaxis] * self.m[np.newaxis, :]
        # F_strength = m_product * inv_dist  # (N, N)

        # TODO: Threshold forces for distances above a threshold

        unit_vect = delta / self.dist[:, :, np.newaxis]
        unit_vect[self.dist == 0] = 0

        attract_mask, repel_mask = self._get_force_signs()

        # Ensure repel mask includes distances below r0
        repel_mask = torch.logical_or(repel_mask, self.dist < self.r0)

        # Initialize force strengths
        F_strength = torch.zeros_like(self.dist)

        # Apply conditional force rules
        F_strength[attract_mask] = (
            -self.k_attract * m_product[attract_mask] / self.dist[attract_mask]
        )
        F_strength[repel_mask] = (
            self.k_repel * m_product[repel_mask] / self.dist[repel_mask]
        )

        F_vec = F_strength[:, :, np.newaxis] * unit_vect
        # return F_strength, F_vec, delta
        total_force = torch.sum(F_vec, axis=1)
        return total_force

    def kinetic_step(self, softening=1e-2):
        forces = self._compute_forces()
        acceleration = forces / (self.m[:, np.newaxis] + softening)
        self.v += acceleration * self.dt
        # add damping
        self.v *= 0.99
        self.pos_vec += self.v * self.dt

        # Wrap positions back into the box

        if self.bouncing:
            # Bouncing walls
            for dim in [0, 1]:  # x and y
                mask_low = self.pos_vec[:, dim] < 0
                mask_high = self.pos_vec[:, dim] > self.box_size[dim]

                # Reflect position
                self.pos_vec[mask_low, dim] = 0
                # self.pos_vec[mask_high, dim] = self.box_size[dim]
                self.pos_vec[mask_high, dim] = (
                    2 * self.box_size[dim] - self.pos_vec[mask_high, dim]
                )

                # Reverse velocity on bounce
                self.v[mask_low | mask_high, dim] *= -1
        else:
            self.pos_vec = self.pos_vec % self.box_size

        self.positions = self.pos_vec.view((self.w, self.h, 2))

        self.time += 1

    def ca_step(self):
        new_world = self.world.clone()  # (H * W, 2), copy of the world
        neighbors = self._get_neighbors()

        for n_i, neigh in enumerate(neighbors):
            neigh_count = self.world[neigh].sum(axis=0)

            max_state = neigh_count.argmax().item()

            if neigh_count.sum() != 0:
                new_world[n_i] = torch.roll(self.dirac, max_state)

            # if neigh_count[0] > neigh_count[1]:
            #     new_world[n_i] = torch.tensor([1, 0])
            # elif neigh_count[0] < neigh_count[1]:
            #     new_world[n_i] = torch.tensor([0, 1])

            # Get bigger if you are surrounded by the same state
            if (neigh_count[0] > 0) and (neigh_count[0] == len(neigh)):
                self.m[n_i] += 1 * torch.tensor(self.growth).int()
                # new_world[n_i, 0] = 0
            if (neigh_count[1] > 0) and (neigh_count[1] == len(neigh)):
                self.m[n_i] += 1 * torch.tensor(self.growth).int()
                # new_world[n_i, 1] = 0

            if self.m[n_i] > self.mass_max:
                self.m[n_i] = 1
                new_world[n_i, :] = 0

        return new_world

    def step(self):
        """
        Makes one step of the automaton
        """
        self.kinetic_step()

        if self.time > self.ca_time:
            self.time = 0

            self.world = self.ca_step()

        # self.m[self.world.sum(dim=1) > 0] = 1

    def step_old(self):
        """
        Makes one step of the automaton
        """
        new_world = torch.zeros_like(self.world)  # (H,W,3), copy of the world

        # n_l = self.world.roll(-1, dims=0)
        # n_r = self.world.roll(1, dims=0)

        # n_a = self.world.roll(-1, dims=1)
        # n_b = self.world.roll(1, dims=1)

        for y in range(self.h):
            for x in range(self.w):  # loop over all cells
                sum_neigh1 = 0  # initialize the neighbour sum
                sum_neigh2 = 0
                for dx, dy in [
                    (1, 1),
                    (1, 0),
                    (1, -1),
                    (0, 1),
                    (0, -1),
                    (-1, -1),
                    (-1, 0),
                    (-1, 1),
                ]:  # Iterate over Neighbors
                    # for dx,dy in [(1,0),(0,1),(0,-1),(-1,0)]: # Iterate over Neighbors (4)
                    sum_neigh1 += (
                        self.world[(y + dy) % self.h, (x + dx) % self.w, 0] == 1
                    )  # State 1
                    sum_neigh2 += (
                        self.world[(y + dy) % self.h, (x + dx) % self.w, 1] == 1
                    )  # State 2
                if self.world[y, x].sum() == 0:  # Current cell is dead
                    if sum_neigh1 > sum_neigh2:  # Condition for it to become S1
                        new_world[y, x, 0] = 1
                    elif sum_neigh1 < sum_neigh2:  # Condition for it to stay S0
                        new_world[y, x, 1] = 1
                elif self.world[y, x, 0] == 1:  # Current cell is S1
                    if sum_neigh2 > sum_neigh1 + 1:
                        new_world[y, x, 1] = 1
                    else:
                        new_world[y, x, 0] = 1
                else:
                    if sum_neigh1 > sum_neigh2 + 1:
                        new_world[y, x, 0] = 1
                    else:
                        new_world[y, x, 1] = 1

        self.time += 1

        self.world = new_world

    def reset(self):
        """
        Resets the automaton to the initial state.
        """
        self._worldmap = torch.zeros_like(self._worldmap)
        self.time = 0

        self._new_world()

    def process_event(self, event, camera=None):
        """
        DEL -> resets the automaton
        UP -> increase the number of starting cells by 1 (max 20)
        DOWN -> decrease the number of starting cells by 1 (min 2)
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE or event.key == pygame.K_DELETE:
                self.reset()
                self.draw()

            if event.key == pygame.K_UP:
                self.n_active = torch.min(torch.tensor([self.n_active + 1, 20]))
            if event.key == pygame.K_DOWN:
                self.n_active = torch.max(torch.tensor([self.n_active - 1, 2]))

    def get_color_world(self):
        """
        Return colorized sliced world

        Returns : (3,H,W) tensor of floats
        """

        colorize = torch.zeros((self.h, self.w, 3), dtype=torch.float)  # (H,W,3)

        colorize[:, :, :] = self.color_dict[0]
        # colorize[self.world == 2] = self.color_dict[2]
        colorize[self.world[..., 0].bool()] = self.color_dict[1]
        colorize[self.world[..., 1].bool()] = self.color_dict[2]

        return colorize.permute(2, 0, 1)

    def draw(self):
        """
        Draws the current state of the automaton, using arbitrary coloration which looks nice.
        """
        # self._worldmap=self.get_color_world() # (3,H,W)
        # self._worldmap = self.world.view(-1, self.n_states)
        self._worldmap = self.world

    @property
    def worldsurface(self):

        surf = pygame.Surface((self.el_size * self.w, self.el_size * self.w))
        surf.fill("black")

        # x_pos, y_pos = self.positions.view(-1, 2).t().numpy().astype(int)
        x_pos, y_pos = (self.positions.view(-1, 2) * self.scale).t().int()

        cell_states = self._worldmap

        for x, y, state, mass in zip(x_pos, y_pos, cell_states, self.m):
            # xy_array = np.array([x*self.scale[0], y*self.scale[1]]).astype(int)
            # xy_array = np.array([(x+1)*pos_scale, (y+1)*pos_scale]).astype(int)
            if state.sum() == 0:
                pygame.draw.circle(surf, "white", (x, y), self.radius + mass.int())
            elif state[0] == 1:
                pygame.draw.circle(surf, "red", (x, y), self.radius + mass.int())
            else:
                pygame.draw.circle(surf, "blue", (x, y), self.radius + mass.int())
        return surf

    # @property
    # def worldmap(self):
    #     return self._worldmap.view(-1, self.n_states)
