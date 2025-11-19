import torch
import torch.nn as nn
import torch.nn.functional as F



class AgentNetwork(nn.Module):
    """
    Agent network following NQMIX paper architecture

    Architecture (same as QMIX for critic):
    1. Input encoder: 64-dim FC layer
    2. Recurrent layer: GRU with 64-dim hidden state
    3. Critic head: |U|-dim FC layer (outputs Q-value for each action dimension)
    4. Actor head: 64-dim FC layer with ReLU -> action-dim FC layer
    """
    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            hidden_dim: int = 64,
            action_low: float = -0.4,
            action_high: float = 0.4
        ):
        """
        Initialize agent network

        Args:
            obs_dim: Dimension of agent's observation space
            action_dim: Dimension of agent's action space
            hidden_dim = GRU hidden state dimension (paper uses 64)
            action_low = Lower bound of action space 
            action_high = Upper bound of action space 
        """
        super(AgentNetwork, self).__init__()

        # Store dimensions
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # Store action space bounds for proper scaling
        self.action_low = action_low
        self.action_high = action_high
        self.action_scale = (action_high - action_low) / 2.0 # 0.4
        self.action_bias = (action_high + action_low) / 2.0  # 0.0

        # ==============================================
        # SHARED ENCODER + GRU (same as QMIX)
        # ==============================================

        # Paper: "64 dimensional fully-connected layer"
        # Input: concatenation of observation and last action
        # Purpose: Encode current state information before feeding to GRU
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)

        # Paper: "GRU recurrent layer"
        # Purpose: Maintain action-observation history τ = (o_0, u_0, ..., o_t)
        # Hidden state encodes the entire history into a fixed-size vector
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # ==============================================
        # Critic HEAD (same as QMIX)
        # ==============================================

        # Paper: "|U|-dimensional fully conncected layer"
        # Note: in continuous action space, we need Q(τ, u) not Q(τ) for all u
        # So we take hidden state + action as input, output scalar Q-value
        # This differs slightly from discrete QMIX but is standard for continuous
        self.critic_fc = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.critic_out = nn.Linear(hidden_dim, 1)

        # ==============================================
        # Actor HEAD (new in NQMIX)
        # ==============================================

        # Paper: "each agent has its local policy that is a 64-dimensional
        #         fully-connected layer with ReLU activation before it"
        # Paper: "Each local policy takes the hidden state of GRU as input"

        # Actor: hidden_dim -> 64 -> action_dim
        # First layer: 64-dim with ReLU (as specified in paper)
        self.actor_fc = nn.Linear(hidden_dim, 64)

        # Second layer: outputs action (one value per action dimension)
        self.actor_out = nn.Linear(64, action_dim)


    def forward(
            self,
            obs: torch.Tensor,
            last_action: torch.Tensor,
            hidden: torch.Tensor,
            current_action: torch.Tensor= None
        ):
        """
        Forward pass through the agent network.

        Flow:
        1. Encode: [obs, last_action] -> FC -> ReLU -> encoded_input
        2. Recurrence: GRU(encoded_input, old_hidden) -> new_hidden (this is τ)
        3. Actor: new_hidden -> FC(64) -> ReLU -> FC(action_dim) -> tanh -> scaled_action
        4. Critic: [new_hidden, action] -> FC -> ReLU -> FC(1) -> Q-value

        Args:
            obs: Current observation [batch, obs_dim]
            last_action: Previous action [batch, action_dim]
            hidden: GRU hidden state encoding history τ_{t-1} [batch, hidden_dim]
            current_action: Action to evaluate Q-value (optional) [batch, action_dim]

        Returns:
            q_value: Q_a(τ_t, current_action) or None if current_action not provided
            action: μ_a(τ_t) - deterministic action from policy
            new_hidden: Updated hidden state τ_t [batch, hidden_dim]
        """

        # ==============================================
        # STEP 1: Encode observation and last action (same as QMIX)
        # ==============================================
        # Concatenate obs and last_action to give network context
        x = torch.cat([obs, last_action], dim=-1) #  [batch, obs_dim + action_dim]

        # Pass through 64-dim FC layer with ReLU
        x = F.relu(self.fc1(x))

        # ==============================================
        # STEP 2: Update GRU hidden state (same as QMIX)
        # ==============================================
        # GRU updates hidden state: h_t = GRU(x_t, h_{t-1})
        # new_hidden now encodes the history: τ_t = [o_0, u_0, ..., o_t, u_{t-1}]
        new_hidden = self.gru(x, hidden) # [batch, 64]

        # ==============================================
        # STEP 3: Actor - Generate action from policy μ_a(τ_t|θ) (NEW in NQMIX)
        # ==============================================
        # Paper: "64-dimensional fully-connected layer with ReLU activation"
        actor_x = F.relu(self.actor_fc(new_hidden))

        # Output layer produces raw action logits
        action_normalized = torch.tanh(self.actor_out(actor_x)) # # [batch, action_dim] in [-1, 1]

        # Scale from [-1, 1] to [action_low, action_high]
        # For Humanoid: [-1, 1] -> [-0.4, 0.4]
        action = action_normalized * self.action_scale + self.action_bias

        # ==============================================
        # STEP 4: Critic - Evaluate Q_a(τ_t, u) if action provided (same as QMIX)
        # ==============================================
        q_value = None
        if current_action is not None:
            # Concatenate hidden state (history τ) with action to evaluate
            critic_x = torch.cat([new_hidden, current_action], dim=-1)

            # Pass through critic network
            critic_x = F.relu(self.critic_fc(critic_x)) # [batch, 64]
            q_value = self.critic_out(critic_x)

        return q_value, action, new_hidden
    
    def init_hidden(self, batch_size: int= 1):
        """
        Initialize GRU hidden state (represents empty history at episode start).

        Args:
            batch_size: Number of parallel episodes

        Returns:
            Zero tensor [batch_size, hidden_dim]
        """
        return torch.zeros(batch_size, self.hidden_dim)
    

