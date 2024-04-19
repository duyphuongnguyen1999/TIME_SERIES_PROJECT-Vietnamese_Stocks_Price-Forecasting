import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn


class ModelManager:
    def __init__(self, model, train_loader, val_loader=None, lr=0.001, patience=100):
        """
        A class to manage the training and validation of a model.
        Args:
            model (nn.Module): The model to train
            train_loader (DataLoader): The DataLoader for training data
            val_loader (DataLoader): The DataLoader for validation data
            lr (float): The learning rate for the optimizer
            patience (int): The number of epochs to wait before early stopping
        """
        # Assign self object
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.criterion = nn.L1Loss()  #  Mean absolute error (MAE)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train(self, num_epochs, save_dir="."):
        """
        A function to train the model.
        Args:
            num_epochs (int): The number of epochs to train
            save_dir (str): The directory to save the model
        """
        # Create a directory to save model
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"best-{self.model.__class__.__name__}.pth")

        # Train model
        for epoch in range(num_epochs):
            start_time = time.time()
            self.model.train()  # Set the model to training mode
            total_train_loss = 0

            for inputs, targets in self.train_loader:
                # Forward pass
                outputs = self.model(inputs)  # Train model with input data
                loss = self.criterion(outputs, targets)  # Calculate loss
                total_train_loss += loss.item()  # Calculate total_train_loss

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            avg_train_loss = total_train_loss / len(self.train_loader)

            # Validate the model
            val_loss = self.evaluate(loader=self.val_loader)

            # Check for early stopping
            if self.early_stopping(val_loss, save_path):
                print(f"Early stopping at epoch {epoch + 1}")
                return

            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"time: {int(time.time() - start_time)}s, "
                f"loss: {avg_train_loss:.4f}, "
                f"val_loss: {val_loss:.4f}"
            )

        self.load_model(save_path)

    def evaluate(self, loader):
        """
        A function to evaluate the model.
        Args:
            loader (DataLoader): The DataLoader for validation data
        Returns:
            float: The average loss of the model
        """
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0

        # Calculate gradient
        with torch.no_grad():
            for inputs, targets in loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        return avg_loss

    def early_stopping(self, val_loss, save_path):
        """
        A function to check for early stopping.
        Args:
            val_loss (float): The loss of the model on the validation set
            save_path (str): The path to save the model
        Returns:
            bool: True if early stopping is triggered, False otherwise
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.save_model(save_path)
        else:
            self.counter += 1
        return self.counter >= self.patience

    def save_model(self, save_path):
        """
        A function to save the model.
        Args:
            save_path (str): The path to save the model
        """
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, load_path):
        """
        A function to load the model.
        Args:
            load_path (str): The path to load the model
        """
        self.model.load_state_dict(torch.load(load_path))
        print(f"Model loaded from {load_path}")

    def predict(self, input_data):
        """
        A function to predict the output of the model.
        Args:
            input_data (torch.Tensor or DataLoader): The input data to predict
        Returns:
            torch.Tensor: The predicted output
        """
        self.model.eval()  # Set the model to evaluation mode

        if isinstance(input_data, DataLoader):
            # If input_data is a DataLoader, iterate through batches and concatenate predictions
            predictions = []
            with torch.no_grad():
                for inputs, _ in input_data:
                    outputs = self.model(inputs)
                    predictions.append(outputs)
            predictions = torch.cat(predictions, dim=0)
        else:
            # Assume input_data is a single input tensor
            with torch.no_grad():
                predictions = self.model(input_data).unsqueeze(0)

        return predictions

    def plot(
        self,
        y,
        yhat,
        feature_names=None,
        save_dir=".",
        save_plots=True,
        num_elements=None,
    ):
        """
        A function to plot the predicted output.
        Args:
            y (torch.Tensor): The true output
            yhat (torch.Tensor): The predicted output
            feature_names (list): The names of the features
            save_dir (str): The directory to save the plots
            save_plots (bool): Whether to save the plots
            num_elements (int): The number of elements to plot
        Returns:
            None
        """
        if feature_names is None:
            feature_names = [f"Feature {i + 1}" for i in range(y.shape[2])]

        if num_elements is not None:
            y = y[:num_elements]
            yhat = yhat[:num_elements]

        for feature_index, feature_name in enumerate(feature_names):
            plt.figure(figsize=(10, 5))

            plt.plot(y[:, :, feature_index].flatten(), label="y", linestyle="-")
            plt.plot(yhat[:, :, feature_index].flatten(), label="y_hat", linestyle="--")

            plt.title(feature_name)
            plt.xlabel("Time Step")
            plt.ylabel("Values")
            plt.legend()

            if save_plots:
                # Create the save directory if it doesn't exist
                os.makedirs(
                    os.path.join(save_dir, self.model.__class__.__name__), exist_ok=True
                )

                # Save the plot
                save_path = os.path.join(
                    save_dir, self.model.__class__.__name__, f"{feature_name}.png"
                )
                plt.savefig(save_path)

            plt.show()
            plt.close()  # Close the plot to avoid overlapping in saved images
