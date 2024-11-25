from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # Define the MLP layers 
        self.fc1 = nn.Linear(n_track * 4, 128) # Input size is n_track * 4 (2 for left and 2 for right) 
        self.fc2 = nn.Linear(128, 64) 
        self.fc3 = nn.Linear(64, n_waypoints * 2) # Output size is n_waypoints * 2

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # Flatten the input tensors 
        x = torch.cat([track_left, track_right], dim=-1) # Shape: (b, n_track, 4) 
        x = x.view(x.size(0), -1) # Shape: (b, n_track * 4) 
        
        # Pass through the MLP layers 
        x = torch.relu(self.fc1(x)) 
        x = torch.relu(self.fc2(x)) 
        x = self.fc3(x) 
        
        # Reshape the output to (b, n_waypoints, 2) 
        waypoints = x.view(x.size(0), self.n_waypoints, 2) 
        return waypoints


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # Learned query embeddings for waypoints
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # Linear layer to encode the input lane boundaries 
        self.input_proj = nn.Linear(2, d_model) 
        
        # Transformer decoder layer 
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8) 
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=6) 
        
        # Output projection to predict waypoints 
        self.output_proj = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """

        # Concatenate left and right track boundaries 
        track = torch.cat([track_left, track_right], dim=1) # Shape: (b, 2 * n_track, 2) 
        
        # Encode the input lane boundaries 
        track_encoded = self.input_proj(track) # Shape: (b, 2 * n_track, d_model) 
        
        # Get the query embeddings for waypoints 
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, track.size(0), 1) # Shape: (n_waypoints, b, d_model) 
        
        # Apply the transformer decoder 
        tgt = torch.zeros_like(query_embed) # Shape: (n_waypoints, b, d_model) 
        memory = track_encoded.permute(1, 0, 2) # Shape: (2 * n_track, b, d_model) 
        output = self.transformer_decoder(tgt, memory) # Shape: (n_waypoints, b, d_model) 
        
        # Project the output to predict waypoints 
        waypoints = self.output_proj(output.permute(1, 0, 2)) # Shape: (b, n_waypoints, 2) 
        return waypoints
        #raise NotImplementedError
    
class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        # Define the CNN layers 
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) 
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) 
        
        # Define the fully connected layers 
        self.fc1 = nn.Linear(256 * 6 * 8, 512) 
        self.fc2 = nn.Linear(512, n_waypoints * 2)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Pass through the CNN layers 
        x = torch.relu(self.conv1(x)) 
        x = torch.relu(self.conv2(x)) 
        x = torch.relu(self.conv3(x)) 
        x = torch.relu(self.conv4(x))
        
        # Flatten the tensor 
        x = x.view(x.size(0), -1) 
        
        # Pass through the fully connected layers 
        x = torch.relu(self.fc1(x)) 
        x = self.fc2(x) 
        
        # Reshape the output to (b, n_waypoints, 2) 
        waypoints = x.view(x.size(0), self.n_waypoints, 2) 
        return waypoints
        #raise NotImplementedError


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        print("Hello")
        print(model_path)
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
