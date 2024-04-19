import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

class PoseNetwork:
    def __init__(self) -> None:
        self.mlp_regressor = MLPRegressor(hidden_layer_sizes=(200,100, 50), activation='relu', solver='adam', alpha=0.001, max_iter=1000, random_state=42, early_stopping=True, validation_fraction=0.1)
        self.scaler = StandardScaler()

    def convertdata(self,data):
        '''extract z axis data'''
        data_z = data[:,6]
        return data_z



    def create_lagged_features(self,data, lag):
        '''Create lagged features for time series forecasting'''
        X = []
        y = []
        for i in range(len(data) - lag):
            X.append(data[i:i+lag])
            y.append(data[i+lag])
        return np.array(X), np.array(y)

    def pose_train(self,pose):
        #generate train,test, time and full reference z axis data
        train_data = self.convertdata(pose)
        #train and predict 
        lag = 10  # Number of lagged features
        X_train, y_train = self.create_lagged_features(train_data, lag)

        # Scale the input features
        X_train_scaled = self.scaler.fit_transform(X_train)
        # Train the MLPRegressor with early stopping
        self.mlp_regressor.fit(X_train_scaled, y_train)



    def pose_predict(self,prev_poses):
        prev_z = prev_poses[:,6]
        prev_z = prev_z.reshape(1,-1)
        prev_z_scaled = self.scaler.transform(prev_z)
        z = self.mlp_regressor.predict(prev_z_scaled)
        new_pose = np.copy(prev_poses[-1])
        new_pose[6] = z
        return new_pose
    
