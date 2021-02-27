import turtle
wn = turtle.Screen()
alex = turtle.Turtle() 
alex.speed(10) #initialize the turtle

#Specify the parameters
sides=20
dist = 10
angle = (sides-2)*180/sides

#Draw the shape as per the specified parameters
for i in range(sides):
    for _ in range(sides):
        alex.forward(dist)
        alex.left(180-angle)
    alex.left(360/sides)