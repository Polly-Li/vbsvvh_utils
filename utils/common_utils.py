

 # general purposed functions
def create_folder(file_name):
    import os
    if not os.path.exists(file_name):
        os.makedirs(file_name, exist_ok=True)
