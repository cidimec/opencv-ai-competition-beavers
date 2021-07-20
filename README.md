# opencv-ai-competition-beavers
## List of codes and explanation
1. To run two detection models with different input image sizes:

![alt text](https://github.com/cidimec/opencv-ai-competition-beavers/blob/main/Images/fig2_models.png)

```
python3 2_two_models.py
```

2. Obtain the 3D position (x, y, z) and calculates the distance between the objects:

![alt text](https://github.com/cidimec/opencv-ai-competition-beavers/blob/main/Images/fig3_distance.png)

```
python3 3_calculate_distance.py
```

3. Implementing probability model:

![alt text](https://github.com/cidimec/opencv-ai-competition-beavers/blob/main/Images/fig4_probability.png)

```
python3 4_probability.py
```

4. Sending information to Ubidots and alerting by Telegram:

![alt text](https://github.com/cidimec/opencv-ai-competition-beavers/blob/main/Images/Ubidots.png)

![alt text](https://github.com/cidimec/opencv-ai-competition-beavers/blob/main/Images/Telegram.png)

```
python3 5_Ubidots&Telegram.py
```

The information can be seen in this link:
https://stem.ubidots.com/app/dashboards/public/dashboard/a_HfJthGvk0VDJVFnpFlyOZOJNSB6jOFylS68S7gY4k?nonavbar=true

5. Deploying Jetson Nano to activate alarms and display to show probability

```
python3 6_Jetson_Alarms.py
```

## Mounts for OAK-D
### Tripod:

![alt text](https://github.com/cidimec/opencv-ai-competition-beavers/blob/main/OAK%20mounts/Tripod/Animations/Tripod_Assembly.gif)

![alt text](https://github.com/cidimec/opencv-ai-competition-beavers/blob/main/OAK%20mounts/Tripod/Animations/Animation.gif)
