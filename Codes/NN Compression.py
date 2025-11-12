import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from copy import deepcopy
import time
import networkx as nx
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
import traceback
import math
from sklearn.metrics import r2_score


class NeuralNetworkCompressor:
    def __init__(self, model, compression_ratio=0.5, Tmin=0.01, alpha=0.99, 
                 PERTURB=0.0005, STOP=1e-5, T=80, seed=42, device='auto'):
        """
        Initialize the neural network compressor using deterministic annealing.
        
        Args:
            model: PyTorch neural network model
            compression_ratio: Target compression ratio for hidden layers
            Tmin: Minimum temperature for annealing
            alpha: Annealing rate
            PERTURB: Perturbation factor for centroids
            STOP: Convergence threshold
            T: Initial temperature
            seed: Random seed for reproducibility
            device: Device to use ('cuda', 'cpu', or 'auto')
        """
        self.model = deepcopy(model)
        self.compression_ratio = compression_ratio
        self.Tmin = Tmin
        self.alpha = alpha
        self.PERTURB = PERTURB
        self.STOP = STOP
        self.T = T
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Set device
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Try to move model to device with error handling
        try:
            self.model.to(self.device)
            print("Model successfully moved to device.")
        except Exception as e:
            print(f"Error moving model to device: {e}")
            print("Falling back to CPU.")
            self.device = torch.device("cpu")
            self.model.to(self.device)
        
        # Store original model parameters for comparison
        self.original_params = {name: param.data.clone() for name, param in self.model.named_parameters()}
        self.compressed_params = {}
        self.compression_stats = {}
        self.association_matrices = {}
        self.compressed_model = None
        
    def calculate_distortion(self, X, Y):
        """
        Calculate the squared Euclidean distance between each point in X and each centroid in Y.
        Uses GPU acceleration when available.
        
        Args:
            X (torch.Tensor): Data matrix of shape (M, N)
            Y (torch.Tensor): Centroid matrix of shape (K, N)
            
        Returns:
            torch.Tensor: Distance matrix of shape (M, K)
        """
        X_sum_sq = torch.sum(X**2, dim=1, keepdim=True)
        Y_sum_sq = torch.sum(Y**2, dim=1, keepdim=True).T
        D = X_sum_sq + Y_sum_sq - 2 * X @ Y.T
        return D
    
    def run_deterministic_annealing(self, X, K):
        """
        Run deterministic annealing clustering algorithm on GPU.
        
        Args:
            X (np.ndarray): Data matrix of shape (M, N)
            K (int): Number of clusters
            
        Returns:
            tuple: (centroids Y, association matrix P)
        """
        M, N = X.shape
        
        # Convert to PyTorch tensor and move to device
        try:
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        except Exception as e:
            print(f"Error creating tensor on device: {e}")
            print("Falling back to CPU for this operation.")
            X_tensor = torch.tensor(X, dtype=torch.float32, device='cpu')
        
        # Px is the weight for each data point, assuming uniform weights
        Px = torch.full((M, 1), 1 / M, device=X_tensor.device)
        
        # Initialize centroids to the weighted mean of data
        initial_mean = (Px.T @ X_tensor).reshape(1, -1)
        Y = initial_mean.repeat(K, 1)
        
        T = self.T
        while T >= self.Tmin:
            L_old = float('inf')
            while True:
                # Calculate distortion matrix
                D = self.calculate_distortion(X_tensor, Y)
                
                # Calculate probability matrix using softmax
                D_bar = D - torch.min(D, dim=1, keepdim=True).values
                num = torch.exp(-D_bar / T)
                den = torch.sum(num, dim=1, keepdim=True)
                P = num / den
                
                # Update centroids
                Py = P.T @ Px
                Py[Py == 0] = 1e-10  # Avoid division by zero
                Y = (P.T @ (X_tensor * Px)) / Py.reshape(-1, 1)
                
                # Add small random perturbation
                Y += self.PERTURB * torch.rand(*Y.shape, device=Y.device)
                
                # Calculate loss function
                L = -T * (Px.T @ torch.log(torch.sum(torch.exp(-D_bar / T), dim=1, keepdim=True)))
                
                # Check for convergence
                if torch.abs(L - L_old) < self.STOP:
                    break
                    
                L_old = L
                
            # Decrease temperature
            T *= self.alpha
            
        # Convert results back to numpy for compatibility with PyTorch layers
        return Y.cpu().numpy(), P.cpu().numpy()
    
    def compress_layer(self, layer, input_dim=None):
        """
        Compress a single layer using deterministic annealing.
        
        Args:
            layer: PyTorch layer (Linear or Conv2d)
            input_dim: Input dimension for the layer (if needed)
            
        Returns:
            tuple: (compressed_layer, association_matrix)
        """
        if isinstance(layer, nn.Linear):
            return self.compress_linear_layer(layer)
        elif isinstance(layer, nn.Conv2d):
            return self.compress_conv_layer(layer)
        else:
            # For non-compressible layers, return the original layer and identity matrix
            return layer, None
    
    def compress_linear_layer(self, layer):
        """
        Compress a linear layer using deterministic annealing, treating bias as a special neuron.
        """
        # Get weight matrix and bias
        weight_matrix = layer.weight.data.cpu().numpy()  # (out_features, in_features)
        out_features, in_features = weight_matrix.shape

        print(weight_matrix.shape)
        
        # Create augmented weight matrix including bias as a special neuron
        if layer.bias is not None:
            bias = layer.bias.data.cpu().numpy()  # (out_features,)
            # Add a column for the bias neuron (which always outputs 1)
            augmented_matrix = np.hstack([weight_matrix, bias.reshape(-1, 1)])
        else:
            # If no bias, just use the weight matrix
            augmented_matrix = weight_matrix
        
        # Determine number of clusters based on compression ratio
        K = max(1, int(out_features * self.compression_ratio))
        
        # Skip compression if K equals out_features (compression_ratio=1)
        if K >= out_features:
            return layer, None
        
        # Run deterministic annealing on the augmented matrix
        centroids, P = self.run_deterministic_annealing(augmented_matrix, K)
        
        # Get the device and dtype from the original layer
        device = layer.weight.device
        dtype = layer.weight.dtype
        
        # Split centroids into weights and bias
        if layer.bias is not None:
            # Last column of centroids represents the bias
            compressed_weights = centroids[:, :-1]
            compressed_bias = centroids[:, -1]
        else:
            compressed_weights = centroids
            compressed_bias = None
        
        # Create compressed linear layer
        compressed_layer = nn.Linear(in_features, K).to(device)
        compressed_layer.weight.data = torch.tensor(compressed_weights, device=device, dtype=dtype)
        
        if compressed_bias is not None:
            compressed_layer.bias.data = torch.tensor(compressed_bias, device=device, dtype=dtype)
        else:
            compressed_layer.bias = None
            
        return compressed_layer, P
        
    def compress_conv_layer(self, layer):
        """
        Compress a convolutional layer using deterministic annealing, treating bias as a special neuron.
        """
        # Get weight tensor and bias
        weight_tensor = layer.weight.data.cpu().numpy()  # (out_channels, in_channels, kernel_h, kernel_w)
        out_channels, in_channels, kernel_h, kernel_w = weight_tensor.shape
        
        # Reshape weight tensor to 2D matrix
        weight_matrix = weight_tensor.reshape(out_channels, -1)  # (out_channels, in_channels * kernel_h * kernel_w)
        
        # Create augmented weight matrix including bias as a special neuron
        if layer.bias is not None:
            bias = layer.bias.data.cpu().numpy()  # (out_channels,)
            # Add a column for the bias neuron (which always outputs 1)
            augmented_matrix = np.hstack([weight_matrix, bias.reshape(-1, 1)])
        else:
            # If no bias, just use the weight matrix
            augmented_matrix = weight_matrix
        
        # Determine number of clusters based on compression ratio
        K = max(1, int(out_channels * self.compression_ratio))
        
        # Skip compression if K equals out_channels (compression_ratio=1)
        if K >= out_channels:
            return layer, None
        
        # Run deterministic annealing on the augmented matrix
        centroids, P = self.run_deterministic_annealing(augmented_matrix, K)
        
        # Get the device and dtype from the original layer
        device = layer.weight.device
        dtype = layer.weight.dtype
        
        # Split centroids into weights and bias
        if layer.bias is not None:
            # Last column of centroids represents the bias
            compressed_weights = centroids[:, :-1]
            compressed_bias = centroids[:, -1]
        else:
            compressed_weights = centroids
            compressed_bias = None
        
        # Reshape compressed weights back to tensor form
        compressed_weights_tensor = compressed_weights.reshape(K, in_channels, kernel_h, kernel_w)
        
        # Create compressed convolutional layer
        compressed_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=K,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
            bias=(compressed_bias is not None)
        ).to(device)
        
        compressed_layer.weight.data = torch.tensor(compressed_weights_tensor, device=device, dtype=dtype)
        
        if compressed_bias is not None:
            compressed_layer.bias.data = torch.tensor(compressed_bias, device=device, dtype=dtype)
        else:
            compressed_layer.bias = None
            
        return compressed_layer, P
    

    def _compress_holistically(self, current_layer, next_layer):
        """
        (Private Helper) Compresses a hidden layer by clustering a representation 
        of its neurons that includes both incoming and outgoing weights.
        Also measures the distortion.
        """
        print(f"--- Starting Holistic Compression on {current_layer} and {next_layer} ---")
        device = current_layer.weight.device
        dtype = current_layer.weight.dtype
        
        # 1. IDENTIFY WEIGHTS
        W1 = current_layer.weight.data.cpu().numpy()
        b1 = current_layer.bias.data.cpu().numpy()
        W2 = next_layer.weight.data.cpu().numpy()
        
        # print(W2)
        # print()
        # print(b1)
        # print()
     
        # 2. COMBINE MATRICES
        W2_T = W2.T
        # print("W2")
        # print(W2_T)
        # print()
        
        augmented_W1 = np.hstack([W1, b1.reshape(-1, 1)])
        combined_matrix = np.concatenate([augmented_W1, W2_T], axis=1)
        print(f"Created combined matrix of shape: {combined_matrix.shape}")
    
        # 3. AGGREGATE
        num_neurons = combined_matrix.shape[0]
        K = max(1, int(num_neurons * self.compression_ratio))
        centroids, P = self.run_deterministic_annealing(combined_matrix, K)
        print(f"Aggregated into {K} centroids.")

        P_T = P.T
        # print("\n",P_T)
        
        # 5. SEPARATE (Done before distortion measurement to get new weights)
        new_aug_w1 = centroids[:, :augmented_W1.shape[1]]
        new_w2_t = centroids[:, augmented_W1.shape[1]:]
        new_w1 = new_aug_w1[:, :-1]
        new_b1 = new_aug_w1[:, -1]
        new_w2 = new_w2_t.T


        # Original Weights multiplied by the association matrix
        # p_sum = np.sum(P_T, axis = 1)
        # p_sum = p_sum.reshape(-1,1)
        # # print(p_sum)
        # print()

        # print(P.shape)
        
        # W1_M_P = ((P_T)@W1)/p_sum
        # b1_M_P = ((P_T)@b1)/p_sum
        # W2_M_P = (P_T@W2_T)/p_sum
        # W2_M_P = W2_M_P.T

        # new_w1 = W1_M_P
        # new_b1 = b1_M_P
        # new_w2 = W2_M_P
        
        # print(new_w1)
        # print()
        # print("W2_M_P")
        # print(W2_M_P)
        # print()
        # print(new_b1)
        # print()
        # print("new_w2")
        # print(new_w2)
        # print()

        # 4. MEASURE DISTORTION (now compares original vs reconstructed weights)
        P_tensor = torch.tensor(P, device=device, dtype=dtype)
        
        # Reconstruct the original weight matrices from the new compressed ones
        new_w1_tensor = torch.tensor(new_w1, device=device, dtype=dtype)
        W1_recon_tensor = P_tensor @ new_w1_tensor

        # print(W1_recon_tensor)
        # print()
        
        new_w2_tensor = torch.tensor(new_w2, device=device, dtype=dtype)
        # Note: The reconstruction for W2 uses P.T
        W2_recon_tensor = new_w2_tensor @ P_tensor.T
        
        # Original weight tensors
        W1_tensor = torch.tensor(W1, device=device, dtype=dtype)
        W2_tensor = torch.tensor(W2, device=device, dtype=dtype)
        
        # Calculate dissimilarity using the provided function.
        # The diagonal of the resulting distance matrix gives the squared distance
        # between corresponding original and reconstructed neuron weights.
        D1_matrix = self.calculate_distortion(W1_tensor, W1_recon_tensor)
        dissimilarity_W1 = torch.sum(torch.diag(D1_matrix))
        
        D2_matrix = self.calculate_distortion(W2_tensor, W2_recon_tensor)
        dissimilarity_W2 = torch.sum(torch.diag(D2_matrix))
        
        print(f"Dissimilarity for W1 (incoming weights): {dissimilarity_W1.item():.4f}")
        print(f"Dissimilarity for W2 (outgoing weights): {dissimilarity_W2.item():.4f}")
    
        # 6. CREATE NEW LAYERS
        new_current_layer = nn.Linear(in_features=new_w1.shape[1], out_features=new_w1.shape[0], bias=True)
        new_current_layer.to(device)
        new_current_layer.weight.data = torch.tensor(new_w1, device=device, dtype=dtype)
        new_current_layer.bias.data = torch.tensor(new_b1, device=device, dtype=dtype)
        
        new_next_layer = nn.Linear(in_features=new_w2.shape[1], out_features=new_w2.shape[0], bias=(next_layer.bias is not None))
        new_next_layer.to(device)
        new_next_layer.weight.data = torch.tensor(new_w2, device=device, dtype=dtype)
        if new_next_layer.bias is not None:
            new_next_layer.bias.data = next_layer.bias.data.clone()
            
        print("--- Holistic Compression Complete ---")
        return new_current_layer, new_next_layer, P

    
    # def compress_model(self):
    #     """
    #     Compress the entire neural network model, adjusting subsequent layers.
    #     Now compresses all hidden layers including the first one (fc1),
    #     while not compressing the input layer and output layer.
    #     """
    #     compressed_model = deepcopy(self.model)
        
    #     # Try to move compressed model to device with error handling
    #     try:
    #         compressed_model.to(self.device)
    #         print("Model successfully moved to device and Compressing...")
    #     except Exception as e:
    #         print(f"Error moving compressed model to device: {e}")
    #         print("Falling back to CPU.")
    #         self.device = torch.device("cpu")
    #         compressed_model.to(self.device)
    #         self.model.to(self.device)
        
    #     self.association_matrices = {}
    #     prev_P = None
    #     prev_layer_name = None

        
    #     # Get a list of all modules in the model
    #     modules = list(compressed_model.named_modules())

    #     # First pass: identify compressible layers
    #     # Collect all linear and conv layers
    #     linear_conv_layers = []
    #     for name, module in modules:
    #         if isinstance(module, (nn.Linear, nn.Conv2d)):
    #             linear_conv_layers.append((name, module))
        
    #     # The first layer is the input layer, the last layer is the output layer
    #     input_layer_name = linear_conv_layers[0][0]
    #     output_layer_name = linear_conv_layers[-1][0]
        
    #     print(f"Input layer: {input_layer_name}")
    #     print(f"Output layer: {output_layer_name}")
        
    #     # Second pass: compress layers and adjust dimensions
    #     for name, module in modules:
    #         if isinstance(module, (nn.Linear, nn.Conv2d)):
    #             # Get parent module and attribute name
    #             path = name.split('.')
    #             parent = compressed_model
    #             for p in path[:-1]:
    #                 parent = getattr(parent, p)
    #             attr_name = path[-1]
    #             layer = getattr(parent, attr_name)
                
    #             # If this layer follows a compressed layer, adjust its input dimensions
    #             if prev_P is not None:
    #                 # Convert P to tensor and move to device
    #                 try:
    #                     P_tensor = torch.tensor(prev_P, dtype=torch.float32, device=self.device)
    #                 except Exception as e:
    #                     print(f"Error creating P tensor on device: {e}")
    #                     print("Falling back to CPU for this operation.")
    #                     P_tensor = torch.tensor(prev_P, dtype=torch.float32, device='cpu')
                    
    #                 if isinstance(layer, nn.Linear):
    #                     # Adjust input dimensions for linear layer
    #                     original_weight = layer.weight.data
    #                     new_weight = original_weight @ P_tensor
                        
    #                     # Create new linear layer with adjusted input dimensions
    #                     new_in_features = prev_P.shape[1]
    #                     new_layer = nn.Linear(new_in_features, layer.out_features, bias=(layer.bias is not None))
    #                     new_layer.to(self.device)
    #                     new_layer.weight.data = new_weight
                        
    #                     if layer.bias is not None:
    #                         new_layer.bias.data = layer.bias.data.clone()
                        
    #                     # Replace the layer
    #                     setattr(parent, attr_name, new_layer)
    #                     layer = new_layer
                
    #             # Compress the current layer if it's not the output layer
    #             # Now we compress all layers except the output layer
    #             if name != output_layer_name:
    #                 compressed_layer, P = self.compress_layer(layer)
    #                 setattr(parent, attr_name, compressed_layer)
                    
    #                 # Store the association matrix for the next layer
    #                 prev_P = P
    #                 self.association_matrices[name] = P
    #                 print(f"Compressed layer {name}: {layer} -> {compressed_layer}")
    #             else:
    #                 # Reset prev_P for non-compressible layers
    #                 prev_P = None
    #                 print(f"Skipped layer {name} (output layer)")

    #      # Store compressed parameters and stats
    #     self.compressed_params = {name: param.data.clone() for name, param in compressed_model.named_parameters()}
    #     self.calculate_compression_stats()
        
    #     # Store the compressed model
    #     self.compressed_model = compressed_model
        
    #     # Return the compressed model
    #     return compressed_model


    def compress_model(self):
        """
        Compresses the entire model by iteratively applying the holistic, 
        two-layer compression strategy.
        """
        compressed_model = deepcopy(self.model)
        
        # Helper functions to get/set modules by their string name (e.g., 'net.0')
        def get_module_by_name(model, name):
            path = name.split('.')
            module = model
            for p in path:
                # Check for Sequential indexing
                if p.isdigit():
                    module = module[int(p)]
                else:
                    module = getattr(module, p)
            return module
    
        def set_module_by_name(model, name, new_module):
            path = name.split('.')
            parent = model
            for p in path[:-1]:
                if p.isdigit():
                    parent = parent[int(p)]
                else:
                    parent = getattr(parent, p)
            
            attr_name = path[-1]
            if attr_name.isdigit():
                parent[int(attr_name)] = new_module
            else:
                setattr(parent, attr_name, new_module)
    
        # Get a list of the names of all compressible layers
        compressible_layer_names = []
        for name, module in compressed_model.named_modules():
            if isinstance(module, nn.Linear):
                compressible_layer_names.append(name)
        
        if len(compressible_layer_names) < 2:
            print("Not enough compressible layers to apply holistic compression.")
            return compressed_model
    
        print(f"Found compressible layers: {compressible_layer_names}")
    
        # Iterate through the layers in overlapping pairs, e.g., (L0, L2), then (L2, L4)
        # We stop before the last layer, as it cannot be a "current_layer" with a "next_layer".
        for i in range(1,len(compressible_layer_names) - 1):
        # for i in range(1,2):
            current_layer_name = compressible_layer_names[i]
            next_layer_name = compressible_layer_names[i+1]
            
            print(f"\nProcessing pair: ({current_layer_name}, {next_layer_name})")
            
            # Get the actual layer modules from the potentially modified compressed_model
            current_layer = get_module_by_name(compressed_model, current_layer_name)
            next_layer = get_module_by_name(compressed_model, next_layer_name)
    
            # Perform the holistic compression on the current pair
            new_current_layer, new_next_layer, P = self._compress_holistically(current_layer, next_layer)
            
            # Replace these two layers in the compressed_model
            set_module_by_name(compressed_model, current_layer_name, new_current_layer)
            set_module_by_name(compressed_model, next_layer_name, new_next_layer)
            
            self.association_matrices[current_layer_name] = P
            print(f"Updated layers '{current_layer_name}' and '{next_layer_name}' in the model.")
    
        # Store final model details
        print("\nModel compression complete.")
        self.compressed_params = {name: param.data.clone() for name, param in compressed_model.named_parameters()}
        self.calculate_compression_stats()
        self.compressed_model = compressed_model
        
        return compressed_model

    
        
    def calculate_compression_stats(self):
        """
        Calculate and store compression statistics.
        """
        original_params = sum(p.numel() for p in self.original_params.values())
        compressed_params = sum(p.numel() for p in self.compressed_params.values())
        
        self.compression_stats = {
            'original_parameters': original_params,
            'compressed_parameters': compressed_params,
            'compression_ratio': 1 - (compressed_params / original_params),
            'parameter_reduction': original_params - compressed_params
        }
    
    def evaluate_model(self, model, test_loader):
        """
        Evaluate the model on a test dataset using GPU.
        
        Args:
            model: PyTorch model to evaluate
            test_loader: DataLoader for test dataset
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        model.eval()
        model.to(self.device)
        
        total_loss = 0.0
        total_samples = 0
        criterion = nn.MSELoss().to(self.device)  # Use MSE for regression
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets).item() * inputs.size(0)
                total_loss += loss
                total_samples += inputs.size(0)
        
        avg_loss = total_loss / total_samples
        
        return {
            'loss': avg_loss
        }
    
    def compare_models(self, test_loader):
        """
        Compare the original and compressed models on a test dataset using GPU.
        
        Args:
            test_loader: DataLoader for test dataset
            
        Returns:
            dict: Dictionary containing comparison metrics
        """
        # Evaluate original model
        original_results = self.evaluate_model(self.model, test_loader)
        
        # Evaluate compressed model
        if self.compressed_model is None:
            compressed_model = self.compress_model()
        else:
            compressed_model = self.compressed_model
            
        compressed_results = self.evaluate_model(compressed_model, test_loader)
        
        # Calculate differences
        loss_diff = compressed_results['loss'] - original_results['loss']
        
        return {
            'original_loss': original_results['loss'],
            'compressed_loss': compressed_results['loss'],
            'loss_difference': loss_diff,
            **self.compression_stats
        }
    
    def visualize_compression(self, layer_name=None):
        """
        Visualize the compression of a specific layer or the entire model.
        
        Args:
            layer_name: Name of the layer to visualize (if None, visualize all layers)
        """
        if layer_name:
            # Visualize a specific layer
            if layer_name in self.original_params and layer_name in self.compressed_params:
                original_weights = self.original_params[layer_name].cpu().numpy()
                compressed_weights = self.compressed_params[layer_name].cpu().numpy()
                
                plt.figure(figsize=(12, 6))
                
                plt.subplot(1, 2, 1)
                plt.imshow(original_weights, cmap='viridis')
                plt.title(f'Original Weights - {layer_name}')
                plt.colorbar()
                
                plt.subplot(1, 2, 2)
                plt.imshow(compressed_weights, cmap='viridis')
                plt.title(f'Compressed Weights - {layer_name}')
                plt.colorbar()
                
                plt.tight_layout()
                plt.show()
            else:
                print(f"Layer {layer_name} not found in model parameters.")
        else:
            # Visualize compression statistics
            stats = self.compression_stats
            if stats:
                plt.figure(figsize=(10, 6))
                
                plt.subplot(1, 2, 1)
                categories = ['Original', 'Compressed']
                values = [stats['original_parameters'], stats['compressed_parameters']]
                plt.bar(categories, values)
                plt.title('Parameter Count')
                plt.ylabel('Number of Parameters')
                
                plt.subplot(1, 2, 2)
                plt.pie([stats['compression_ratio'], 1 - stats['compression_ratio']], 
                        labels=['Compressed', 'Remaining'], autopct='%1.1f%%')
                plt.title('Compression Ratio')
                
                plt.tight_layout()
                plt.show()
    

    def model_to_graph(self, model):
        """
        Convert a neural network model to a graph representation.
        Includes bias neurons as special nodes.
        Creates all edges without filtering by weight.
        Now includes the input layer with dynamic size.
        
        Args:
            model: PyTorch model
            
        Returns:
            tuple: (networkx.DiGraph, dict) - Graph and node positions
        """
        G = nx.DiGraph()
        node_positions = {}
        layer_counts = {}
        layer_nodes = {}
        
        # Add input layer
        input_layer_name = "input"
        
        # Determine input size dynamically from the first layer
        first_layer = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                first_layer = module
                break
        
        if first_layer is not None:
            input_size = first_layer.in_features
            print(f"Detected input size: {input_size}")
        else:
            # Default fallback
            input_size = 3  # Default for your model
            print(f"Using default input size: {input_size}")
        
        layer_counts[input_layer_name] = input_size
        
        # First pass: count nodes per layer
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                layer_counts[name] = module.out_features
            elif isinstance(module, nn.Conv2d):
                layer_counts[name] = module.out_features
        
        # Print layer information for debugging
        print("\nLayer information for model:")
        for name, count in layer_counts.items():
            print(f"{name}: {count} nodes")
        
        # Add input layer nodes
        layer_idx = 0
        nodes = []
        for i in range(input_size):
            node_id = f"{input_layer_name}_{i}"
            G.add_node(node_id, layer=input_layer_name, index=i, type='input')
            nodes.append(node_id)
            
            # Position nodes in a grid
            x = layer_idx * 2
            y = (i - input_size/2) * 0.5
            node_positions[node_id] = (x, y)
        
        layer_nodes[input_layer_name] = nodes
        layer_idx += 1
        
        # Add a single bias node for the entire network
        bias_node_id = "bias"
        G.add_node(bias_node_id, layer="bias", type='bias', value=1.0)
        x_bias = 0  # Position bias node to the left of all layers
        y_bias = 0
        node_positions[bias_node_id] = (x_bias, y_bias)
        
        # Second pass: create nodes and edges for the rest of the model
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Create nodes for this layer
                nodes = []
                if isinstance(module, nn.Linear):
                    num_nodes = module.out_features
                else:  # Conv2d
                    num_nodes = module.out_features
                
                for i in range(num_nodes):
                    node_id = f"{name}_{i}"
                    G.add_node(node_id, layer=name, index=i, type='neuron')
                    nodes.append(node_id)
                    
                    # Position nodes in a grid
                    x = layer_idx * 2
                    y = (i - num_nodes/2) * 0.5
                    node_positions[node_id] = (x, y)
                
                layer_nodes[name] = nodes
                layer_idx += 1
        
        # Create edges between consecutive layers
        layer_names = list(layer_nodes.keys())
        print(f"\nLayer names: {layer_names}")
        
        for i in range(len(layer_names) - 1):
            current_layer = layer_names[i]
            next_layer = layer_names[i+1]
            
            print(f"\nCreating edges from {current_layer} to {next_layer}")
            
            if current_layer == "input":
                # Create edges from input layer to first hidden layer
                first_hidden_module = dict(model.named_modules())[layer_names[i+1]]
                
                if isinstance(first_hidden_module, nn.Linear):
                    weights = first_hidden_module.weight.data.cpu().numpy()
                    bias = first_hidden_module.bias.data.cpu().numpy() if first_hidden_module.bias is not None else None
                    
                    print(f"Weight matrix shape: {weights.shape}")
                    print(f"Max absolute weight: {np.max(np.abs(weights))}")
                    
                    # Connect input neurons to hidden neurons
                    for j, next_node in enumerate(layer_nodes[next_layer]):
                        for k, current_node in enumerate(layer_nodes[current_layer]):
                            # Check if indices are within bounds
                            if j < weights.shape[0] and k < weights.shape[1]:
                                weight = weights[j, k]
                                # Add all edges regardless of weight
                                G.add_edge(current_node, next_node, weight=weight)
                    
                    # Connect bias to all hidden neurons
                    if bias is not None:
                        print(f"Bias shape: {bias.shape}")
                        print(f"Max absolute bias: {np.max(np.abs(bias))}")
                        
                        for j, next_node in enumerate(layer_nodes[next_layer]):
                            if j < bias.shape[0]:
                                bias_weight = bias[j]
                                # Connect from the single bias node
                                G.add_edge(bias_node_id, next_node, weight=bias_weight)
            else:
                # Create edges between regular layers using NEXT layer's weights
                next_module = dict(model.named_modules())[next_layer]
            
                if isinstance(next_module, nn.Linear):
                    weights = next_module.weight.data.cpu().numpy()
                    bias = next_module.bias.data.cpu().numpy() if next_module.bias is not None else None
            
                    print(f"Weight matrix shape: {weights.shape}")
                    print(f"Max absolute weight: {np.max(np.abs(weights))}")
            
                    # weights[j, k]: from current_layer neuron k -> next_layer neuron j
                    for j, next_node in enumerate(layer_nodes[next_layer]):
                        for k, current_node in enumerate(layer_nodes[current_layer]):
                            if j < weights.shape[0] and k < weights.shape[1]:
                                w = weights[j, k]
                                G.add_edge(current_node, next_node, weight=w)
            
                    # Bias of NEXT layer connects to its neurons
                    if bias is not None:
                        print(f"Bias shape: {bias.shape}")
                        print(f"Max absolute bias: {np.max(np.abs(bias))}")
                        for j, next_node in enumerate(layer_nodes[next_layer]):
                            if j < bias.shape[0]:
                                bw = bias[j]
                                G.add_edge("bias", next_node, weight=bw)
            
                elif isinstance(next_module, nn.Conv2d):
                    # Same idea: use next_module, not current_module
                    weights = next_module.weight.data.cpu().numpy()
                    bias = next_module.bias.data.cpu().numpy() if next_module.bias is not None else None
                    out_c, in_c, kh, kw = weights.shape
            
                    print(f"Weight tensor shape: {weights.shape}")
                    print(f"Max absolute weight: {np.max(np.abs(weights))}")
            
                    for j, next_node in enumerate(layer_nodes[next_layer]):
                        for k, current_node in enumerate(layer_nodes[current_layer]):
                            if j < out_c and k < in_c:
                                avg_w = np.mean(weights[j, k])
                                G.add_edge(current_node, next_node, weight=avg_w)
            
                    if bias is not None:
                        print(f"Bias shape: {bias.shape}")
                        print(f"Max absolute bias: {np.max(np.abs(bias))}")
                        for j, next_node in enumerate(layer_nodes[next_layer]):
                            if j < bias.shape[0]:
                                bw = bias[j]
                                G.add_edge("bias", next_node, weight=bw)
        
        print(f"\nGraph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G, node_positions
    
    def draw_neural_network(self, G, pos, title="Neural Network", 
                           edge_percentage=0.01, min_edges=100, max_edges=5000,
                           min_alpha=0.1, max_width=3.0, min_width=0.5):
        """
        Draw neural network with proper sign coloring and edge thickness.
        - Blue = positive weights
        - Red = negative weights
        - Thickness ∝ |weight|
        - If params <= 1000 → draw all edges, else sample subset
        """
        start_time = time.time()
        plt.figure(figsize=(14, 10))
    
        # Extract edges and weights
        edges = list(G.edges())
        weights = np.array([G[u][v]['weight'] for u, v in edges])
        abs_weights = np.abs(weights)
    
        # Determine if full or partial visualization
        total_edges = len(edges)
        num_params = total_edges  # edges ~ params for visualization
        if num_params <= 1000:
            selected_idx = np.arange(total_edges)
            print(f"Drawing all {total_edges} edges.")
        else:
            num_edges_to_draw = max(int(total_edges * edge_percentage), min_edges)
            num_edges_to_draw = min(num_edges_to_draw, total_edges)
            print(f"Sampling {num_edges_to_draw}/{total_edges} edges ({100*num_edges_to_draw/total_edges:.1f}%)")
            selected_idx = np.random.choice(total_edges, size=num_edges_to_draw, replace=False)
    
        selected_edges = [edges[i] for i in selected_idx]
        selected_weights = weights[selected_idx]
        selected_abs = abs_weights[selected_idx]
    
        # Normalize for visual scaling
        max_w = np.max(selected_abs) if np.max(selected_abs) > 0 else 1
        normalized = selected_abs / max_w
    
        edge_widths = min_width + (max_width - min_width) * normalized
        edge_alphas = min_alpha + (1.0 - min_alpha) * normalized
        edge_colors = ['blue' if w > 0 else 'red' for w in selected_weights]
    
        # Draw nodes by type
        input_nodes = [n for n in G.nodes if G.nodes[n].get('type') == 'input']
        neuron_nodes = [n for n in G.nodes if G.nodes[n].get('type') == 'neuron']
        bias_nodes = [n for n in G.nodes if G.nodes[n].get('type') == 'bias']
    
        nx.draw_networkx_nodes(G, pos, nodelist=input_nodes, node_size=200, node_color='lightcoral', alpha=0.9)
        nx.draw_networkx_nodes(G, pos, nodelist=neuron_nodes, node_size=200, node_color='lightgreen', alpha=0.9)
        nx.draw_networkx_nodes(G, pos, nodelist=bias_nodes, node_size=200, node_color='lightblue', alpha=0.9)
    
        # Draw edges
        for (u, v), c, w, a in zip(selected_edges, edge_colors, edge_widths, edge_alphas):
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=c, width=w, alpha=a, arrows=True, arrowsize=10)
    
        # Label layers
        layer_positions = {}
        for node, data in G.nodes(data=True):
            layer = data.get('layer', '')
            layer_positions.setdefault(layer, []).append(pos[node])
    
        for layer, pts in layer_positions.items():
            x = np.mean([p[0] for p in pts])
            y = max([p[1] for p in pts]) + 0.8
            plt.text(x, y, layer, ha='center', fontsize=12, fontweight='bold')
    
        plt.title(title, fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        duration = time.time() - start_time
        print(f"Rendered in {duration:.2f}s")
        plt.show()
        return duration
    
    def visualize_networks(self, edge_percentage=0.01, min_edges=100, max_edges=5000,
                          min_alpha=0.1, max_width=3.0, min_width=0.5):
        """
        Visualize both the original and compressed neural networks.
        Includes bias neurons as special nodes.
        Only draws a percentage of edges with the highest absolute weights.
        Now includes the input layer with dynamic size.
        
        Args:
            edge_percentage: Percentage of edges to draw (0-1)
            min_edges: Minimum number of edges to draw
            max_edges: Maximum number of edges to draw
            min_alpha: Minimum opacity for edges (smallest weights)
            max_width: Maximum width for edges (largest weights)
            min_width: Minimum width for edges (smallest weights)
        """
        # Create compressed model if not already created
        if self.compressed_model is None:
            self.compress_model()
        
        # Convert models to graphs
        try:
            print("Creating graph for original model...")
            original_graph, original_pos = self.model_to_graph(self.model)
        except Exception as e:
            print(f"Error creating graph for original model: {e}")
            traceback.print_exc()
            return
        
        try:
            print("Creating graph for compressed model...")
            compressed_graph, compressed_pos = self.model_to_graph(self.compressed_model)
        except Exception as e:
            print(f"Error creating graph for compressed model: {e}")
            traceback.print_exc()
            return
        
        # Draw original network
        try:
            print("Drawing original network...")
            original_duration = self.draw_neural_network(
                original_graph, 
                original_pos, 
                title="Original Neural Network",
                edge_percentage=edge_percentage,
                min_edges=min_edges,
                max_edges=max_edges,
                min_alpha=min_alpha,
                max_width=max_width,
                min_width=min_width
            )
        except Exception as e:
            print(f"Error drawing original network: {e}")
            traceback.print_exc()
            original_duration = 0
        
        # Draw compressed network
        try:
            print("Drawing compressed network...")
            compressed_duration = self.draw_neural_network(
                compressed_graph, 
                compressed_pos, 
                title="Compressed Neural Network",
                edge_percentage=edge_percentage,
                min_edges=min_edges,
                max_edges=max_edges,
                min_alpha=min_alpha,
                max_width=max_width,
                min_width=min_width
            )
        except Exception as e:
            print(f"Error drawing compressed network: {e}")
            traceback.print_exc()
            compressed_duration = 0
        
        # Print compression statistics
        print("\nCompression Statistics:")
        for key, value in self.compression_stats.items():
            print(f"{key}: {value}")
        
        # Print timing statistics
        print("\nTiming Statistics:")
        print(f"Original graph drawing time: {original_duration:.2f} seconds")
        print(f"Compressed graph drawing time: {compressed_duration:.2f} seconds")
        print(f"Total drawing time: {original_duration + compressed_duration:.2f} seconds")

    def visualize_trained_network(self, model, max_nodes_per_layer=200):
        """
        Visualize a fully-connected feedforward network from its learned weights.
        Colors represent sign (red=negative, blue=positive),
        and edge width represents magnitude.
        
        Args:
            model: torch.nn.Module with nn.Linear layers
            max_nodes_per_layer: limit nodes for clarity in large layers
        """
        layers = [module for module in model.modules() if isinstance(module, torch.nn.Linear)]
        n_layers = len(layers) + 1
        layer_sizes = [layers[0].in_features] + [l.out_features for l in layers]
    
        # Normalize node count for drawing clarity
        display_sizes = [min(s, max_nodes_per_layer) for s in layer_sizes]
        
        # Setup figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        ax.set_title("Learned Neural Network Structure", fontsize=14, weight='bold')
    
        # Position nodes
        layer_positions = []
        x_spacing = 1 / (len(display_sizes) - 1)
        for i, n_nodes in enumerate(display_sizes):
            y_spacing = 1 / (n_nodes + 1)
            layer_positions.append([(i * x_spacing, 1 - (j + 1) * y_spacing) for j in range(n_nodes)])
        
        # Draw edges
        for i, layer in enumerate(layers):
            W = layer.weight.data.cpu().numpy()
            # Clip large layers for clarity if truncated
            W = W[:display_sizes[i+1], :display_sizes[i]]
            
            max_w = np.max(np.abs(W)) + 1e-8
            for out_idx, out_pos in enumerate(layer_positions[i+1]):
                for in_idx, in_pos in enumerate(layer_positions[i]):
                    w = W[out_idx, in_idx]
                    color = 'blue' if w > 0 else 'red'
                    alpha = np.clip(abs(w) / max_w, 0.1, 1.0)
                    lw = 0.5 + 3 * abs(w) / max_w
                    ax.plot([in_pos[0], out_pos[0]], [in_pos[1], out_pos[1]],
                            color=color, alpha=alpha, linewidth=lw, zorder=1)
    
        # Draw nodes
        for layer in layer_positions:
            for (x, y) in layer:
                circle = plt.Circle((x, y), 0.01, color='lightgray', ec='k', zorder=3)
                ax.add_artist(circle)
    
        # Annotate layers
        for i, size in enumerate(layer_sizes):
            ax.text(i * x_spacing, 1.05, f"Layer {i}\n({size})",
                    ha='center', fontsize=10, weight='bold')
    
        plt.show()


class SingleNN(nn.Module):
    def __init__(self, input_dim = 4, hidden_dim = 10, num_layer = 2, output_dim = 1, dropout = 0.1):
        super(SingleNN, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim),\
                                 nn.SiLU(),\
                                 nn.Linear(hidden_dim, 15),\
                                 nn.SiLU(),\
                                 nn.Linear(15, output_dim))

    def forward(self, x):
        return self.net(x)

criterion  = nn.MSELoss()


df = pd.read_csv("/kaggle/input/single-layer-models/F1_data.csv", header = None)
df = df.drop(index = 0)
df = df.astype(float)

y = torch.tensor(df.iloc[0:, 3].values, dtype = torch.float).unsqueeze(1)
X = torch.tensor(df.iloc[:,0:3].values, dtype = torch.float)

X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.2, random_state = 42)

# Standardize inputs using training statistics
X_mean, X_std = X_train.mean(dim=0, keepdim=True), X_train.std(dim=0, keepdim=True)
X_train_std = (X_train - X_mean) / X_std
X_val_std = (X_val - X_mean) / X_std

# Standardize outputs (helps numerical stability)
y_mean, y_std = y_train.mean(), y_train.std()
y_train_std = (y_train - y_mean) / y_std
y_val_std = (y_val - y_mean) / y_std


train_ds = TensorDataset(X_train_std, y_train_std)
val_ds = TensorDataset(X_val_std, y_val_std)

batch_size = 32
train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(val_ds, batch_size = batch_size, shuffle = True)

print(f"Training samples : {len(train_ds)} | Validation Samples: {len(val_ds)}")


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



# Load the saved weights
try:
    model = torch.load("/kaggle/input/single-layer-models/F1_model_full.pth",map_location=device, weights_only = False)
    print("Model loaded.")
except Exception as e:
    print("Error loading:", e)



# Test Script
# CRs = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
# pert = [0.0005, 0.001, 0.005, 0.01]

CRs = [0.5]
pert = [0.0005,0.001]

results_df = pd.DataFrame(columns=[
     'Comp ratio',
     'Temp',
     'perturb',
     'val_loss',
     'val_r2', # Added R2 column
     'time_seconds'
])
results_file_path = 'compression_test_results.csv'

print("="*50)
print("Starting Hyperparameter Test Run")
print("="*50)

for ratio in CRs:
    for perturb in pert:
        print(f"\n--- Testing: Ratio={ratio}, Perturb={perturb} ---")
        result_data = {
                    'Comp ratio': ratio,
                    'Temp' : 80,
                    'perturb': perturb
        }
        start_time = time.time()

            
        # Evaluate original model
        model.eval()
        original_val_loss = 0.0
        original_all_targets = []
        original_all_predictions = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                original_val_loss += criterion(preds,yb).item() * xb.size(0)
                original_all_targets.append(yb.cpu().numpy())
                original_all_predictions.append(preds.cpu().numpy())
        
        original_avg_loss = original_val_loss / len(test_loader.dataset)
        original_all_targets = np.concatenate(original_all_targets)
        original_all_predictions = np.concatenate(original_all_predictions)
        original_r2 = r2_score(original_all_targets, original_all_predictions)

        print(f'\nOriginal Model loss: {original_avg_loss:.5f}')
        print(f'Original Model R2 Score: {original_r2:.5f}')
        
        # for name, param in model.named_parameters():
        #     if "weight" in name:
        #         print(f"\n{name} shape: {param.shape}")
        #         print(param)  # raw tensor
        
        # Create compressor with CUDA support
        try:
            compressor = NeuralNetworkCompressor(model, test_loader=test_loader, T = 80, PERTURB = perturb,  compression_ratio=ratio, device=device)

        except Exception as e:
            print(f"Error creating compressor: {e}")
            traceback.print_exc()
            # If CUDA fails, try with CPU
            device = torch.device("cpu")
            compressor = NeuralNetworkCompressor(model, test_loader=test_loader, T = 80, PERTURB = perturb,  compression_ratio=ratio, device=device)
        
        # Compress the model
        try:
            compressed_model = compressor.compress_model()
            print("Compressed model created and stored.")
        except Exception as e:
            print(f"Error during compression: {e}")
            traceback.print_exc()
            exit(1)
        
        # Print compressed model architecture
        print("\nCompressed model architecture:")
        for name, param in compressed_model.named_parameters():
            print(f"{name}: {param.shape}")
        
        end_time = time.time()
        duration = end_time - start_time

        # Evaluate the compressed model
        compressed_model.eval()
        comp_val_loss = 0

        comp_all_targets = []
        comp_all_predictions = []
        
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = compressed_model(xb)
                comp_val_loss += criterion(preds,yb).item() * xb.size(0)
                comp_all_targets.append(yb.cpu().numpy())
                comp_all_predictions.append(preds.cpu().numpy())
        
        comp_avg_loss = comp_val_loss / len(test_loader.dataset)
        comp_all_targets = np.concatenate(comp_all_targets)
        comp_all_predictions = np.concatenate(comp_all_predictions)
        comp_r2 = r2_score(comp_all_targets, comp_all_predictions)

        print(f'\nCompressed Model loss: {comp_avg_loss:.5f}')
        print(f'Compressed Model R2 Score: {comp_r2:.5f}')
        
        final_loss = comp_avg_loss
        
        # Compare models
        # try:
        #     comparison = compressor.compare_models(test_loader)
        #     print("\nCompression Statistics:")
        #     for key, value in comparison.items():
        #         print(f"{key}: {value}")
        # except Exception as e:
        #     print(f"Error during model comparison: {e}")
        #     traceback.print_exc()
        
        compressor.visualize_trained_network(model)
        compressor.visualize_trained_network(compressed_model)
        # Visualize compression with optimized edge sampling
        # try:
        #     # For large graphs, use a smaller percentage of edges
        #     compressor.visualize_networks(
        #         edge_percentage=1,  # 0.5% of edges
        #         min_edges=100,        # At least 100 edges
        #         max_edges=2000,       # At most 2000 edges
        #         min_alpha=0.1,        # Minimum opacity
        #         max_width=3.0,        # Maximum edge width
        #         min_width=0.5         # Minimum edge width
        #     )
        # except Exception as e:
        #     print(f"Error during visualization: {e}")
        #     traceback.print_exc()

        result_data['val_loss'] = final_loss
        result_data['val_r2'] = comp_r2 # Storing R2 score
        result_data['time_seconds'] = duration

        new_row_df = pd.DataFrame([result_data])
        results_df = pd.concat([results_df, new_row_df], ignore_index=True)


print("\n--- All test runs complete. ---")
if not results_df.empty:
    results_df.to_csv(results_file_path, index=False)
    print(f"Results saved to '{results_file_path}'")
    print("\nFinal Results Summary:")
    print(results_df)
else:
    print("No results were logged.")
