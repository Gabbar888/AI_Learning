A matplotlib plot is always three nested layers, in 2D or 3D:

```
Figure          ← the whole window/canvas (the page)
  └─ Axes       ← one plot area inside it (a chart)
       └─ artists  ← the actual drawn things (surface, s, arrows, labels)
```
The only thing that makes a plot "3D" is that the Axes is created with projection='3d'. Everything else is the same idea as 2D.


These four's are the irreducible core of any 3D plot — everything in your ax1 block reduces to this:

```python
fig = plt.figure()                                  # 1. the canvas
ax1 = fig.add_subplot(121, projection='3d')         # 2. a 3D axes on it
ax1.plot_surface(X, Y, Z)                            # 3. draw something 3D
plt.show()                                           # 4. render it
```

121 in the add_subplot represents that figure will have 1 row 2 columns and ax1 will be in position 1

Role of Meshgrid function (storing X and Y part of )
X, Y = np.meshgrid([-3, 0, 3], [-3, 0, 3])

X =  [[-3,  0,  3],          Y =  [[-3, -3, -3],
      [-3,  0,  3],                [ 0,  0,  0],
      [-3,  0,  3]]                [ 3,  3,  3]]


```
What	            Method

Title	        ax.set_title(...)	
Axis            labels	ax.set_xlabel / ylabel / zlabel	 
Legend	        ax.legend() (needs label= on plots)	 
Grid	        ax.grid(True, alpha=0.3)	 
Auto-spacing	plt.tight_layout()	 
Save to file	plt.savefig('x.png', dpi=100)	 
Show window	    plt.show()	 
```

Fun Fact : Viewing angle — ax.view_init(elev=30, azim=45) rotates the camera. Search: "matplotlib view_init elevation azimuth"