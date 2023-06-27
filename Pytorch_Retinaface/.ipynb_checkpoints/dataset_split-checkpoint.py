import os
import random
import shutil

class DatasetSplitter:
    def __init__(self, root, destination_dir, split):
        self.root = root
        self.destination_dir = destination_dir
        self.split = split
        #self.train_dir = None
        #self.test_dir = None
        #self.val_dir = None

        
    def split_dataset(self):
        split_total = sum(self.split)
        print(split_total)
        if split_total != 100:
            raise ValueError("Sum of the split percentages is over 100%")

        while True:
            if os.path.exists(self.destination_dir) and os.path.isdir(self.destination_dir):
                train_dir = os.path.join(self.destination_dir, "train_dir")
                os.makedirs(train_dir, exist_ok=True)

                test_dir = os.path.join(self.destination_dir, "test_dir")
                os.makedirs(test_dir, exist_ok=True)
                
                if len(self.split) == 3:
                    val_dir = os.path.join(self.destination_dir, "val_dir")
                    os.makedirs(val_dir, exist_ok=True)

                break
            else:
                os.makedirs(self.destination_dir, exist_ok=True)

        for folder_name in os.listdir(self.root):
            folder_path = os.path.join(self.root, folder_name)
            if os.path.isdir(folder_path):
                train_dir_path = os.path.join(train_dir, folder_name)
                os.makedirs(train_dir_path, exist_ok=True)

                test_dir_path = os.path.join(test_dir, folder_name)
                os.makedirs(test_dir_path, exist_ok=True)

                files = list(os.listdir(folder_path))
                train_files = int(self.split[0] / 100 * len(files))
                test_files = int(self.split[1] / 100 * len(files))

                for file_name in files[:train_files]:
                    source_file_path = os.path.join(folder_path, file_name)
                    train_file_path = os.path.join(train_dir_path, file_name)
                    shutil.move(source_file_path, train_file_path)

                for file_name in files[train_files:(train_files + test_files + 1)]:
                    source_file_path = os.path.join(folder_path, file_name)
                    test_file_path = os.path.join(test_dir_path, file_name)
                    shutil.move(source_file_path, test_file_path)

                if len(self.split) == 3:
                    val_dir_path = os.path.join(val_dir, folder_name)
                    os.makedirs(val_dir_path, exist_ok=True)
                    val_files = int(self.split[2] / 100 * len(files))

                    for file_name in files[train_files + test_files:]:
                        source_file_path = os.path.join(folder_path, file_name)
                        val_file_path = os.path.join(val_dir_path, file_name)
                        shutil.move(source_file_path, val_file_path)
        print("Dataset split: {}% for training and {}% for testing.".format(self.split[0], self.split[1]))
        
        
    def data_dir(self):
        print(len(self.split))

        train_dir = os.path.join(self.destination_dir, "train_dir")
        test_dir = os.path.join(self.destination_dir, "test_dir")
        if len(self.split) == 3:
            val_dir = os.path.join(self.destination_dir, "val_dir")
            return train_dir, test_dir, val_dir
        
        return train_dir, test_dir

  