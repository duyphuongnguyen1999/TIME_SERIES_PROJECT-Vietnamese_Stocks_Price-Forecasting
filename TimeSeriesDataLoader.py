class TimeSeriesDataLoader:
    def __init__(
        self,
        file_path,
        input_size,
        label_size,
        offset,
        train_size,
        val_size,
        time_format="D",
        features_type="ohlc",
        date_column=None,
        batch_size=64,
    ):
        """
        A class to load, preprocess time series data and create Pytorch-DataLoader
        for training, testing and validation.
        Args:
            file_path (str): Path to the CSV file
            input_size (int): Number of time steps used as input features for predicting the next time step
            label_size (int): Number of time steps used as the target for prediction
            offset (int): Number of time steps to shifting
            train_size (float): Percentage of data to use for training
            val_size (float): Percentage of data to use for validation
            time_format (str): Time format to resample time series data
                - 'M': Monthly
                - 'W': Weekly
                - 'D': Daily
                - 'H': Hourly
                - 'T': Minutely
            features_type (str): Type of features to use
                - 'close':
                      Input: Only close price
                      Output: Close price
                - 'ohlc':
                      Input: Open, High, Low, Close
                      Output: Close price
                - 'ohlcv':
                      Input: Open, High, Low, Close, Volume
                      Output: Close price
            date_column (str): Name of the column containing date information
            batch_size (int): Batch size for training
        """
        ##########################################
        #  Run validation to received arguments  #
        ##########################################
        if offset < label_size:
            print(f"Offset will be change from {offset} to {label_size}")
            offset = label_size
        assert (
            input_size > 0
        ), f"input_size should be a positive integer value, but got input_size={input_size}"
        assert (
            label_size > 0
        ), f"label_size should be a positive integer value, but got label_size={label_size}"
        assert (
            train_size > 0 and train_size < 1
        ), f"train_size should be a float value between 0 and 1, but got train_size={train_size}"
        assert (
            val_size > 0 and val_size < 1
        ), f"val_size should be a float value between 0 and 1, but got val_size={val_size}"
        assert (
            batch_size > 0
        ), f"batch_size should be a positive integer value, but got batch_size={batch_size}"

        #######################################
        #  Assign value to object attributes  #
        #######################################
        self.input_size = input_size
        self.label_size = label_size
        self.offset = offset
        self.train_size = train_size
        self.val_size = val_size
        self.batch_size = batch_size
        # Determine feature_name and target_name based on feature_type
        if features_type == "close":
            self.feature_name = ["Close"]
            self.target_name = ["Close"]
        elif features_type == "ohlc":
            self.feature_name = ["Open", "High", "Low", "Close"]
            self.target_name = ["Close"]
        elif features_type == "ohlcv":
            self.feature_name = ["Open", "High", "Low", "Close", "Volume"]
            self.target_name = ["Close"]
        else:
            raise ValueError(
                "Invalid features_type. Choose from 'close', 'ohlc', 'ohlcv'."
            )
        # Determine in_variable and out_variable based on feature_type
        self.in_variable = len(self.feature_name)
        self.out_variable = len(self.target_name)

        ###################
        #  Load the data  #
        ###################
        # Check if file exist or not
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        else:
            self.df = pd.read_csv(file_path)  # Load the data

        #####################################
        #  Preprocessing time serries data  #
        #####################################
        self.__preprocess_time_series(date_column, time_format)

        ##########################################
        #  Create train, validation and dataset  #
        ##########################################
        # Create dataset
        self.X_train, self.y_train = self.__create_dataset(
            start_idx=0, end_idx=int(train_size * len(self.df))
        )
        self.X_val, self.y_val = self.__create_dataset(
            start_idx=int(train_size * len(self.df)),
            end_idx=int((train_size + val_size) * len(self.df)),
        )
        self.X_test, self.y_test = self.__create_dataset(
            start_idx=int((train_size + val_size) * len(self.df)), end_idx=None
        )
        # Print the shape of the dataset
        print(f"{self.X_train.shape = }")
        print(f"{self.y_train.shape = }")
        print(f"{self.X_val.shape = }")
        print(f"{self.y_val.shape = }")
        print(f"{self.X_test.shape = }")
        print(f"{self.y_test.shape = }")

        ####################################
        #  Convert to PyTorch DataLoaders  #
        ####################################
        self.train_loader = self.__create_dataloader(self.X_train, self.y_train)
        self.val_loader = self.__create_dataloader(self.X_val, self.y_val)
        self.test_loader = self.__create_dataloader(self.X_test, self.y_test)

    def __preprocess_time_series(self, date_column, time_format):
        """
        A method to preprocess time series data.
        Args:
            date_column (str): Name of the column containing date information
            time_format (str): Time format to resample time series data
        Returns:
            None
        """
        # Assign date_column
        if date_column is None:
            date_column = self.df.columns[1]
        # Convert date_column to datetime
        self.df[date_column] = pd.to_datetime(self.df[date_column])
        # Set date_column as index
        self.df.set_index(date_column, inplace=True)
        # Resemble time series data
        if time_format in ["M", "W", "D", "H", "T"]:
            self.df = self.df.resample(time_format).agg(
                {
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Volume": "sum",
                    "Close": "last",
                }
            )
        # If time_format is not in the list, raise an error
        else:
            raise ValueError(
                "Invalid time_unit. Choose from 'M'for minute, 'H' for hour, 'D' for day."
            )
        # Drop rows with NaN values
        self.df.dropna(inplace=True)
        # Reset index
        self.df.reset_index(inplace=True)

    def __create_dataset(self, start_idx, end_idx):
        """
        A method to create the dataset for training, testing and validation.
        Args:
            start_idx (int): Start index of the dataset
            end_idx (int): End index of the dataset
        Returns:
            features (np.array): Features of the dataset
            labels (np.array): Labels of the dataset
        """
        if end_idx is None:
            end_idx = len(self.df) - self.label_size - self.offset

        start_idx += self.input_size + self.offset

        features = []
        labels = []
        scaler = MinMaxScaler()

        for idx in range(start_idx, end_idx):
            # Caculate start and end index for features and label slices
            feature_start_idx = idx - self.input_size - self.offset
            feature_end_idx = feature_start_idx + self.input_size

            label_start_idx = idx - 1
            label_end_idx = label_start_idx + self.label_size

            feature = self.df.loc[feature_start_idx:feature_end_idx, self.feature_name]
            label = self.df.loc[label_start_idx:label_end_idx, self.target_name]

            features.append(scaler.fit_transform(feature))
            labels.append(label.to_numpy())
            self.out_features = label.columns
            self.in_features = label.columns

        return np.array(features), np.array(labels)

    def __create_dataloader(self, X, y):
        """
        A function to create the DataLoader for training and validation.
        Args:
            X (np.array): Features of the dataset
            y (np.array): Labels of the dataset
        Returns:
            DataLoader: DataLoader for training and validation.
        """
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
