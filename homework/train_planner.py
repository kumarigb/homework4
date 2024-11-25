import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb
from homework.models import MLPPlanner, TransformerPlanner, CNNPlanner, save_model
from homework.datasets.road_dataset import load_data

def train(model_name: str, transform_pipeline: str, num_workers: int, lr: float, batch_size: int, num_epoch: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(2024)
    np.random.seed(2024)
    
    log_dir = Path("logs") / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)
    
    # Dynamically load the appropriate model 
    if model_name == "mlp_planner": 
        model = MLPPlanner() 
    elif model_name == "transformer_planner": 
        model = TransformerPlanner()
    elif model_name == "cnn_planner": 
        model = CNNPlanner() 
    
    else: raise ValueError(f"Unknown model name: {model_name}")

    model = model.to(device)

    train_loader = load_data("drive_data/train", transform_pipeline=transform_pipeline, return_dataloader=True, num_workers=num_workers, batch_size=batch_size, shuffle=True)
    val_loader = load_data("drive_data/val", transform_pipeline=transform_pipeline, return_dataloader=True, num_workers=num_workers, batch_size=batch_size, shuffle=False)
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    global_step = 0
    for epoch in range(num_epoch):
        model.train()
        train_loss = []
        for batch in train_loader:
            #track_left, track_right, waypoints, waypoints_mask = batch['track_left'].to(device), batch['track_right'].to(device), batch['waypoints'].to(device), batch['waypoints_mask'].to(device)
            if model_name == "cnn_planner": 
                image, waypoints, waypoints_mask = batch['image'].to(device), batch['waypoints'].to(device), batch['waypoints_mask'].to(device) 
                pred_waypoints = model(image) 
            else: 
                track_left, track_right, waypoints, waypoints_mask = batch['track_left'].to(device), batch['track_right'].to(device), batch['waypoints'].to(device), batch['waypoints_mask'].to(device) 
                pred_waypoints = model(track_left, track_right)
            
            optimizer.zero_grad()
            #pred_waypoints = model(track_left, track_right)
            loss = criterion(pred_waypoints, waypoints)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            global_step += 1

        epoch_train_loss = np.mean(train_loss)
        
        model.eval()
        val_loss = []
        with torch.no_grad():
            for batch in val_loader:
                if model_name == "cnn_planner": 
                  image, waypoints, waypoints_mask = batch['image'].to(device), batch['waypoints'].to(device), batch['waypoints_mask'].to(device) 
                  pred_waypoints = model(image) 
                else: 
                  track_left, track_right, waypoints, waypoints_mask = batch['track_left'].to(device), batch['track_right'].to(device), batch['waypoints'].to(device), batch['waypoints_mask'].to(device) 
                  pred_waypoints = model(track_left, track_right) 
                
                loss = criterion(pred_waypoints, waypoints) 
                val_loss.append(loss.item())

        epoch_val_loss = np.mean(val_loss)
        
        print(f"Epoch {epoch + 1:2d} / {num_epoch:2d}: train_loss={epoch_train_loss:.4f} val_loss={epoch_val_loss:.4f}")

        logger.add_scalar("Loss/train", epoch_train_loss, epoch)
        logger.add_scalar("Loss/val", epoch_val_loss, epoch)

    save_model(model)
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--transform_pipeline", type=str, default="state_only")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epoch", type=int, default=40)
    parser.add_argument("--d_model", type=int, default=64) # Add d_model for TransformerPlanner
    args = parser.parse_args()
    train(**vars(args))
