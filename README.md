**Running**
---
Everything you need to evaluate is ready.
For Apple Silicon:
```pip install torch --pre```

Install requirements:
```pip install -r requirements.txt```
```web.py```

Select an image from the ```plant_images/train``` or ```plant_images/test``` to evaluate. The defualt location is ```http://127.0.0.1:8080```

**Training**
---
Requirements:
```
Python 3.10.12
Bing API key for image search
```

For Apple Silicon:
```pip install torch --pre```

Install requirements:
```pip install -r requirements.txt```

First get images from the Wikipedia page of your choice. Make sure to add the Bing API Key to the file:
```get_images.py```

(optional) If the testing set is small, use the move_files.sh script to move 20% of files:
```./move_files plant_images/train plant_images/test```

Train your model using the ResNet18 as the base (trains to 30 epochs):
```train.py```

Example Output for 10 epochs:
```
Epoch 1/10 Loss: 3.4192 Acc: 0.0681
Epoch 2/10 Loss: 2.7664 Acc: 0.2893
Epoch 3/10 Loss: 2.2519 Acc: 0.4182
Epoch 4/10 Loss: 1.8852 Acc: 0.4990
Epoch 5/10 Loss: 1.6163 Acc: 0.5996
Epoch 6/10 Loss: 1.4078 Acc: 0.6488
Epoch 7/10 Loss: 1.2197 Acc: 0.6918
Epoch 8/10 Loss: 1.0836 Acc: 0.7107
Epoch 9/10 Loss: 0.9640 Acc: 0.7348
Epoch 10/10 Loss: 0.8566 Acc: 0.7579
Test Accuracy: 0.2444
```

Metadata and state data is saved and can be used between code and processor types.

Run the webserver and try uploading an images:
```web.py```