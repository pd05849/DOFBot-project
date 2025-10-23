import general_robotics_toolbox as grt
import numpy as np
# :attribute H: A 3 x N matrix containing the direction the joints as unit vectors, one joint per column
# :attribute P: A 3 x (N + 1) matrix containing the distance vector from i to i+1, one vector per column
# :attribute joint_type: A list of N numbers containing the joint type. 0 for rotary, 1 for prismatic, 2 and 3 for mobile

#set up the basis unit vectors
e_x = np.array([[1], [0], [0]])
e_y = np.array([[0], [1], [0]])
e_z = np.array([[0], [0], [1]])

H = np.hstack([e_z, -e_y, -e_y, -e_y, -e_x])

in2m = 2.54
l0 = in2m*(3)
l1 = in2m*(1.125) # base to servo 1
l2 = in2m*(3.25) # servo 1 to 2
l3 = in2m*(3.25) # servo 2 to 3
l4 = in2m*(3.25) # servo 3 to 4
l5 = in2m*(1)

P01 = (l0+l1)*e_z # translation between base frame and 1 frame in base frame
P12 = np.zeros((3, 1)) # translation between 1 and 2 frame in 1 frame
P23 = l2*e_x # translation between 2 and 3 frame in 2 frame
P34 = -l3*e_z # translation between 3 and 4 frame in 3 frame
P45 = np.zeros((3, 1)) # translation between 4 and 5 frame in 4 frame
P5T = -(l4+l5)*e_x # translation between 5 and tool frame in 5 frame

P = np.hstack([P01, P12, P23, P34, P45, P5T])
q_1 = np.array([np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2])

N = [0, 0, 0, 0, 0,]

DOFbot = grt.Robot(H, P, N)

DOFbot_fk = grt.fwdkin(DOFbot, q_1.reshape(-1,1))

# Create a Transform object from the forward kinematics result
T_d = grt.Transform(R=DOFbot_fk.R, p=DOFbot_fk.p)

# Print position vector (rounded)
print("Position vector (meters):")
print(np.around(T_d.p, decimals=3))

# Print rotation matrix
print("\nRotation matrix:")
print(np.around(T_d.R, decimals=3))

# Now let's calculate inverse kinematics using ur_invkin
print("\nCalculating Inverse Kinematics using ur_invkin:")
print("---------------------------------------------")

try:
    # Calculate inverse kinematics using iterative_invkin
    converged, q_sols = grt.iterative_invkin(
        robot=DOFbot,
        desired_pose=T_d,
        q_current=q_1.reshape(-1,1),  # Use current configuration as seed
        max_steps=200,                 # Default max iterations
        Kp=0.3*np.eye(3),             # Position gain
        KR=0.3*np.eye(3),             # Rotation gain
        tol=1e-4                       # Convergence tolerance
    )
    
    print("\nInverse Kinematics Results:")
    print("-------------------------")
    if converged:
        print("Solution found!")
        print("\nJoint angles (radians):")
        print(np.around(q_sols[0], decimals=3))
        
        # Verify the solution
        verify_fk = grt.fwdkin(DOFbot, q_sols[0])
        pos_error = np.linalg.norm(verify_fk.p - T_d.p)
        rot_error = np.linalg.norm(verify_fk.R - T_d.R, ord='fro')
        print(f"\nPosition error: {pos_error:.6f} meters")
        print(f"Rotation error: {rot_error:.6f}")
    else:
        print("Algorithm did not converge to a solution")
        
except Exception as e:
    print(f"\nError in inverse kinematics calculation: {str(e)}")
    print("This might happen if the position is unreachable or if there's no valid solution")

print(q_sols)