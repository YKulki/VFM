
import numpy as np
import pandas as pd
import csv
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time 
import math
import os

initial = time.time()


decimal_places = 10
index = 0
E = 7.143e6

# E_list=np.linspace(6e6,8e6,21)
# E_list=np.array([7.14e6,7.14e6])
# E_list.append(7.143e6)

v = 0.48
# E_star = E/(1-(v**2))
depth_indentation = 1e-4
sphere_radius = 1e-3

# Force = (4/3)*E*(sphere_radius**(1/2))*(depth_indentation**(3/2))
Force=0.3
# force_list=np.linspace(0.2,0.4,21)

a_0 = 3.6 #contact radius

constant = 1 # arbitrary constant value

L = 0.0075 # Length in meters
H = 0.003 # Height in meters
D = L # Depth in meters

df = pd.read_csv('250k.csv')
df.head()
df=df.drop(columns=['ODB Name', 'Step', 'Frame', 'Section Name', 'Material Name', 'Section Point','X','Y','Z','   U-Magnitude'],axis=1)

nodes_displacement=df.loc[df['Part Instance Name'] == 'TISSUE-1']
nodes_displacement=nodes_displacement.drop('Part Instance Name',axis=1)

# print(nodes_displacement.head())
nodes_displacement = np.array(nodes_displacement)

def read_file(file_path):
    nodes = []          # List to store nodal coordinates
    connectivity = []   # List to store element connectivity
    in_node_section = False
    in_element_section = False
    file = open(file_path, 'r')
    Lines = file.readlines()
    flag=0

    for line in Lines:
        line=line.strip()

        if line.startswith('*') and line.startswith('*Part, name=tissue'):
            flag=1
            continue
        elif line.startswith('*Node') and flag==1:
                in_node_section = True
                in_element_section = False
                continue
        elif line.startswith('*Element') and flag==1:
                in_node_section = False
                in_element_section = True
                continue
        elif line.startswith('*') :
            flag=0
            continue
        else:
            if in_node_section and not in_element_section and flag==1:
                values_n = line.split(",")
                node_info = [int(values_n[0])] + [float(values_n[i].strip()) for i in range(1, 4)]
                nodes.append(node_info)
                continue

            elif not in_node_section and in_element_section and flag==1:
                values_e = line.split(",")
                if len(values_e) == 9:
                    element_info = [int(value) for value in values_e]
                    connectivity.append(element_info)
                continue

            else:
                continue

    nodes=np.array(nodes)
    connectivity=np.array(connectivity)

    x_min=np.min(nodes[:,1])
    y_min=np.min(nodes[:,2])
    z_min=np.min(nodes[:,3])

    nodes[:,1]=nodes[:,1]-x_min
    nodes[:,2]=nodes[:,2]-y_min
    nodes[:,3]=nodes[:,3]-z_min

    precision=sys.float_info.epsilon
    nodes[abs(nodes)<precision]=0


    return nodes, connectivity


def calculate_nodes_deformed(nodes,nodes_displacement):
    number_nodes=min(nodes.shape[0],nodes_displacement.shape[0])
    nodes_deformed=np.zeros((number_nodes,4))
    nodes_deformed[:,0]=nodes[:number_nodes,0]
    nodes_deformed[:number_nodes,1:]=nodes[:number_nodes,1:4]+nodes_displacement[:number_nodes,1:4]
    # print((nodes_deformed[0,:]))

    return nodes_deformed


def create_centroids(nodes, connectivity):
    centroids=[]
    for element in connectivity:
        x,y,z=0,0,0
        for i in element[1:]:
            x=x+nodes[i-1,1]
            y=y+nodes[i-1,2]
            z=z+nodes[i-1,3]
        x=x/8
        y=y/8
        z=z/8
        centroids.append([int(element[0]),x,y,z])
    centroids=np.array(centroids)

    return centroids


def calculate_cube_size(centroids,decimal_places):

    x_coordinates = np.sort(np.unique(centroids[:,1]))
    y_coordinates = np.sort(np.unique(centroids[:,2]))
    z_coordinates = np.sort(np.unique(centroids[:,3]))

    #calculating the cube size
    cube_size = [x_coordinates[-1]-x_coordinates[-2], y_coordinates[-1]-y_coordinates[-2],z_coordinates[-1]-z_coordinates[-2]]
    

    # cube_size[0] = round(cube_size[0], decimal_places)
    # cube_size[1] = round(cube_size[1], decimal_places)
    # cube_size[2] = round(cube_size[2], decimal_places)
    # print(cube_size)

    return [cube_size[0], cube_size[1], cube_size[2]]


def map_elements_to_matrix(centroids,cube_size):

    
    # Calculate rows and columns for matrix to fill
    cols = int((max([centroid[1] for centroid in centroids]))/cube_size[0])+1
    rows = int((max([centroid[2] for centroid in centroids]))/cube_size[1])+1
    deps = int((max([centroid[3] for centroid in centroids]))/cube_size[2])+1
    
    print(rows)
    print(cols)
    print(deps)

    matrix = np.zeros((rows, cols, deps), dtype=int)
    for centroid in centroids:
        element_number = centroid[0]
        x = centroid[1]
        y = centroid[2]
        z = centroid[3]
        
        # Calculate indices on matrix
        matrix_x = int(x / cube_size[0])
        matrix_y = int(y / cube_size[1])
        matrix_z = int(z / cube_size[2])
        # Assign element number to matrix
        matrix[matrix_y, matrix_x, matrix_z] = element_number
        #matrix[matrix_x, matrix_y] = element_number
    
    #matrix = np.flip(matrix, axis=0)

    return matrix
 

def map_elements_to_displacement(elements_tensor, displacement_centroids):
    num_elements = elements_tensor.max()

    # Initialize three matrices for X, Y, and Z displacements
    Ux = np.zeros_like(elements_tensor, dtype=float)
    Uy = np.zeros_like(elements_tensor, dtype=float)
    Uz = np.zeros_like(elements_tensor, dtype=float)

    # Fill the displacement matrices with data from displacement_centroids
    for displacement_data in displacement_centroids:
        element_number = displacement_data[0]
        x_displacement = displacement_data[1]
        y_displacement = displacement_data[2]
        z_displacement = displacement_data[3]

        # Find the coordinates of the element in the elements_tensor
        coords = np.argwhere(elements_tensor == element_number)
        y, x, z = coords[0]

        # Map the displacements to their corresponding positions in the matrices
        Ux[y, x, z] = round(x_displacement,decimal_places)
        Uy[y, x, z] = round(y_displacement,decimal_places)
        Uz[y, x, z] = round(z_displacement,decimal_places)

    return Ux, Uy, Uz

def increase_matrix_size(matrix):
    # Get the original dimensions of the matrix
    rows, cols, deps = matrix.shape

    # Create a new matrix with increased size
    new_deps = deps + 2
    new_rows = rows + 2
    new_cols = cols + 2
    new_matrix = np.zeros((new_rows, new_cols, new_deps), dtype=matrix.dtype)

    # Copy the original data from the input matrix to the inner region of the new matrix
    new_matrix[1:-1, 1:-1, 1:-1] = matrix

    # Extend the border of the new matrix by duplicating the values from the first inner layer
    new_matrix[0, :, :] = new_matrix[1, :, :]
    new_matrix[-1, :, :] = new_matrix[-2, :, :]
    new_matrix[:, 0, :] = new_matrix[:, 1, :]
    new_matrix[:, -1, :] = new_matrix[:, -2, :]
    new_matrix[:, :, 0] = new_matrix[:, :, 1]
    new_matrix[:, :, -1] = new_matrix[:, :, -2]

    return new_matrix


def central_differenciation(Ux, Uy, Uz, cube_size):
    rows, cols, deps = len(Ux), len(Ux[0]), len(Ux[0][0])
    dUx_dx_enalarged = np.zeros((rows, cols, deps), dtype=float)
    dUy_dx_enalarged  = np.zeros((rows, cols, deps), dtype=float)
    dUz_dx_enalarged  = np.zeros((rows, cols, deps), dtype=float)
    dUx_dy_enalarged  = np.zeros((rows, cols, deps), dtype=float)
    dUy_dy_enalarged  = np.zeros((rows, cols, deps), dtype=float)
    dUz_dy_enalarged  = np.zeros((rows, cols, deps), dtype=float)
    dUx_dz_enalarged  = np.zeros((rows, cols, deps), dtype=float)
    dUy_dz_enalarged  = np.zeros((rows, cols, deps), dtype=float)
    dUz_dz_enalarged  = np.zeros((rows, cols, deps), dtype=float)

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            for k in range(1, deps-1):
                dUx_dx_enalarged[i, j, k] = round((Ux[i, j+1, k] - Ux[i, j-1, k]) / (2 * cube_size[0]),decimal_places)
                dUy_dx_enalarged[i, j, k] = round((Uy[i, j+1, k] - Uy[i, j-1, k]) / (2 * cube_size[0]),decimal_places)
                dUz_dx_enalarged[i, j, k] = round((Uz[i, j+1, k] - Uz[i, j-1, k]) / (2 * cube_size[0]),decimal_places)
                dUx_dy_enalarged[i, j, k] = round((Ux[i+1, j, k] - Ux[i-1, j, k]) / (2 * cube_size[1]),decimal_places)
                dUy_dy_enalarged[i, j, k] = round((Uy[i+1, j, k] - Uy[i-1, j, k]) / (2 * cube_size[1]),decimal_places)
                dUz_dy_enalarged[i, j, k] = round((Uz[i+1, j, k] - Uz[i-1, j, k]) / (2 * cube_size[1]),decimal_places)
                dUx_dz_enalarged[i, j, k] = round((Ux[i, j, k+1] - Ux[i, j, k-1]) / (2 * cube_size[2]),decimal_places)
                dUy_dz_enalarged[i, j, k] = round((Uy[i, j, k+1] - Uy[i, j, k-1]) / (2 * cube_size[2]),decimal_places) 
                dUz_dz_enalarged[i, j, k] = round((Uz[i, j, k+1] - Uz[i, j, k-1]) / (2 * cube_size[2]),decimal_places)

    dUx_dx = np.zeros((rows-2, cols-2, deps-2), dtype=float)
    dUy_dx = np.zeros((rows-2, cols-2, deps-2), dtype=float)
    dUz_dx = np.zeros((rows-2, cols-2, deps-2), dtype=float)
    dUx_dy = np.zeros((rows-2, cols-2, deps-2), dtype=float)
    dUy_dy = np.zeros((rows-2, cols-2, deps-2), dtype=float)
    dUz_dy = np.zeros((rows-2, cols-2, deps-2), dtype=float)
    dUx_dz = np.zeros((rows-2, cols-2, deps-2), dtype=float)
    dUy_dz = np.zeros((rows-2, cols-2, deps-2), dtype=float)
    dUz_dz = np.zeros((rows-2, cols-2, deps-2), dtype=float)

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            for k in range(1, deps-1):
                dUx_dx[i-1, j-1, k-1] = dUx_dx_enalarged[i, j, k]
                dUy_dx[i-1, j-1, k-1] = dUy_dx_enalarged[i, j, k]
                dUz_dx[i-1, j-1, k-1] = dUz_dx_enalarged[i, j, k]
                dUx_dy[i-1, j-1, k-1] = dUx_dy_enalarged[i, j, k]
                dUy_dy[i-1, j-1, k-1] = dUy_dy_enalarged[i, j, k]
                dUz_dy[i-1, j-1, k-1] = dUz_dy_enalarged[i, j, k]
                dUx_dz[i-1, j-1, k-1] = dUx_dz_enalarged[i, j, k]
                dUy_dz[i-1, j-1, k-1] = dUy_dz_enalarged[i, j, k]
                dUz_dz[i-1, j-1, k-1] = dUz_dz_enalarged[i, j, k]

    return dUx_dx, dUy_dx, dUz_dx, dUx_dy, dUy_dy, dUz_dy, dUx_dz, dUy_dz, dUz_dz


def map_elements_to_centraldiff(elements_tensor, displacement_centroids, dUx_dx, dUy_dx, dUz_dx, dUx_dy, dUy_dy, dUz_dy, dUx_dz, dUy_dz, dUz_dz):
    num_elements = elements_tensor.max()
    rows = len(dUx_dx)
    cols = len(dUx_dx[0])
    deps = len(dUx_dx[0][0])
    # Create a list to store tensor displacement for each element
    tensor_displacement_list = np.zeros((rows,cols,deps), dtype=object)

    for i in range(rows):
        for j in range(cols):
            for k in range(deps):
                arr = np.zeros((3,3), dtype=float)
                arr[0,0] = dUx_dx[i, j, k]
                arr[0,1] = dUx_dy[i, j, k]
                arr[0,2] = dUx_dz[i, j, k]
                arr[1,0] = dUy_dx[i, j, k]
                arr[1,1] = dUy_dy[i, j, k]
                arr[1,2] = dUy_dz[i, j, k]
                arr[2,0] = dUz_dx[i, j, k]
                arr[2,1] = dUz_dy[i, j, k]
                arr[2,2] = dUz_dz[i, j, k]
                tensor_displacement_list[i,j,k] = arr

    return tensor_displacement_list


def calculate_deformation_gradient(tensor_displacement_list, matrix,E,v,undeformed_centroids,cube_size,Force):
    deformation_gradients = np.array(tensor_displacement_list)
    # Identity matrix (3x3)
    rows = len(tensor_displacement_list)
    cols = len(tensor_displacement_list[0])
    deps = len(tensor_displacement_list[0][0])
    I = np.eye(3)

    for i in range(rows):
        for j in range(cols):
           for k in range(deps):
                # if i==0 and j==0 and k==0:
                #     print(tensor_displacement_list[i][j][k])
                deformation_gradients[i][j][k] =deformation_gradients[i][j][k]+ I

    
    deformation_2d_list = []
    for i in range(rows):
        for j in range(cols):
           for k in range(deps):
               element_number = matrix[i,j,k]
               deformation_gradient = deformation_gradients[i][j][k]
               dUx_dx = deformation_gradient[0][0]
               dUx_dy = deformation_gradient[0][1]
               dUx_dz = deformation_gradient[0][2]
               dUy_dx = deformation_gradient[1][0]
               dUy_dy = deformation_gradient[1][1]
               dUy_dz = deformation_gradient[1][2]
               dUz_dx = deformation_gradient[2][0]
               dUz_dy = deformation_gradient[2][1]
               dUz_dz = deformation_gradient[2][2]
               deformation_1d_list = [element_number, dUx_dx, dUx_dy, dUx_dz, dUy_dx, dUy_dy, dUy_dz, dUz_dx, dUz_dy, dUz_dz]
               deformation_2d_list.append(deformation_1d_list)

    
        # deformation_2d_list.sort()
        # filename = 'deformation_gradients_2d.csv'
        # with open(filename, 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     for row in deformation_2d_list:
        #         writer.writerow(row)
    
    pi = []
    IVW_1 = []
    IVW_2 = []



    c_reshaped=np.zeros((3,3))
    e=np.zeros((3,3))
    sigma=np.zeros((3,3))
    pk1=np.zeros((3,3))
    size = 251464
    deformation_2d_array = np.zeros((size, 9), dtype=float)
    pk1_2d_list = np.zeros((size, 9)) #change row number accordingly 
    e_2d_list = np.zeros((size, 9)) #change row number accordingly 
    c_2d_list = np.zeros((size, 9)) #change row number accordingly 
    for i in range(size):
        c_reshaped[0,0] = deformation_2d_list[i][1]
        c_reshaped[0,1] = deformation_2d_list[i][2]
        c_reshaped[0,2] = deformation_2d_list[i][3]
        c_reshaped[1,0] = deformation_2d_list[i][4]
        c_reshaped[1,1] = deformation_2d_list[i][5]
        c_reshaped[1,2] = deformation_2d_list[i][6]
        c_reshaped[2,0] = deformation_2d_list[i][7]
        c_reshaped[2,1] = deformation_2d_list[i][8]
        c_reshaped[2,2] = deformation_2d_list[i][9]
        e = 1/2*(c_reshaped+c_reshaped.T)-np.eye(3)
        I1 = np.trace(e)
        sigma = (E/(1+v))*(e + ((v/(1-2*v))*np.eye(3)*I1))
        inverse_transpose = np.linalg.inv(c_reshaped.T)
        J = np.linalg.det(c_reshaped)
        pk1 = J*np.dot(sigma,inverse_transpose)
        
            

      # Calculate X1 and X2 for each centroid
        X1 = undeformed_centroids[i][1]-L/2
        X2 = undeformed_centroids[i][2]-L/2
        X3 = undeformed_centroids[i][3]


        # Second set of virtual fields and their derivatives

        c=1
        k=1.5
        m=1

        def U_star(d, z):
            t=z/H
            R=a_0*t
            return c*t*(1-(1/(1+np.exp(-k*(d-R)))))
            
            
        def U_star_devX(d, z, x):
            t=z/H
            R=a_0*t
            return -c*t*((k*np.exp(-k*(d-R))*(1+np.exp(-k*(d-R)))**-2))*x*d**(-1/2)

        def U_star_devY(d, z, y):
            t=z/H
            R=a_0*t
            return -c*t*((k*np.exp(-k*(d-R))*(1+np.exp(-k*(d-R)))**-2))*y*d**(-1/2)
            
        def U_star_devZ(d, z):
            t=z/H
            R=a_0*t
            return c/H*(1-(1/(1+np.exp(-k*(d-R)))))

        du1_dX1 = U_star_devX(np.sqrt(X1**2 + X2**2), X3, X1)
        du1_dX2 = U_star_devY(np.sqrt(X1**2 + X2**2), X3, X2)
        du1_dX3 = U_star_devZ(np.sqrt(X1**2 + X2**2), X3)
        du2_dX1 = 0
        du2_dX2 = 0
        du2_dX3 = 0
        du3_dX1 = 0
        du3_dX2 = 0
        du3_dX3 = 0

        # Calculate internal virtual work for second set of virtual fields Π : ∂û/∂X 
        # ivw_1 = (sigma[0, 0] * du1_dX1_1 + sigma[1, 0] * du2_dX1_1 + sigma[0, 1] * du1_dX2_1 + sigma[1, 1] * du2_dX2_1)*(cube_size[0]*cube_size[1]*cube_size[2])

        # Calculate internal virtual work for second set of virtual fields Π : ∂û/∂X 
        ivw_2 = (pk1[0, 0] * du1_dX1 + pk1[1, 0] * du2_dX1 + pk1[2, 0] * du3_dX1 + pk1[0, 1] * du1_dX2 + pk1[1, 1] * du2_dX2 + pk1[2, 1] * du3_dX2 + pk1[0, 2] * du1_dX3 + pk1[1, 2] * du2_dX3 + pk1[2, 2] * du3_dX3)*(cube_size[0]*cube_size[1]*cube_size[2])


        # IVW_1.append(ivw_1)
        IVW_2.append(ivw_2)
          
        
        
    # total_IVW_1 = np.sum(IVW_1)
    total_IVW_2 = np.sum(IVW_2)

    # print(total_IVW_1)
    print(total_IVW_2)
    # Calculate external virtual work for first set of virtual fields 

    # evw_1 = (-2 * force * L)

    # Calculate external virtual work for second set of virtual fields 
    evw_2 = (- Force*0 )
    # print(U_star(0,H))
    # print(evw_1)
    print(evw_2)

    # Cost function
    phi =  np.sqrt((total_IVW_2 - evw_2) ** 2)

    print(phi)
    return phi


def save_3d_matrix_to_csv(matrix_3d, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in matrix_3d[:, :, index]:
            writer.writerow(row)

def save_3d_matrix_enalrged_to_csv(matrix_3d, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in matrix_3d[:, :, index+1]:
            writer.writerow(row)

def save_1d_array_to_csv(array_1d, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Element', '1st Principal Stress'])
        for element_number, value in enumerate(array_1d):
            writer.writerow([element_number + 1, value]) 

def main():
    
    # file_path_undeformed = "250k.inp"

    # decimal_places=8
    # undeformed_nodes, connectivity = read_file(file_path_undeformed)
    # # print(undeformed_nodes)
    # nodes_deformed=calculate_nodes_deformed(undeformed_nodes,nodes_displacement)

    # undeformed_centroids=create_centroids(undeformed_nodes, connectivity)
    # np.save('undeformed_centroids.npy',undeformed_centroids)

    # deformed_centroids=create_centroids(nodes_deformed, connectivity)

    # cube_size=calculate_cube_size(undeformed_centroids,decimal_places)
    # np.save('cube_size.npy',cube_size)

    # matrix = map_elements_to_matrix(undeformed_centroids,cube_size)
    # # print(np.shape(nodes_deformed))
    # np.save('matrix.npy',matrix)


    # displacement_centroids=deformed_centroids
    # displacement_centroids[:,1:4]=deformed_centroids[:,1:4]-undeformed_centroids[:,1:4]
    
    
    # # print(np.shape(displacement_centroids))
    # np.save('displacement_centroids.npy',displacement_centroids)

    # Ux,Uy,Uz=map_elements_to_displacement(matrix,displacement_centroids)

    # Ux_enlarged = increase_matrix_size(Ux)
    # Uy_enlarged = increase_matrix_size(Uy)
    # Uz_enlarged = increase_matrix_size(Uz)

    # # print(np.shape(Ux))

    # dUx_dx, dUy_dx, dUz_dx, dUx_dy, dUy_dy, dUz_dy, dUx_dz, dUy_dz, dUz_dz = central_differenciation(Ux_enlarged, Uy_enlarged, Uz_enlarged, cube_size)
    # # print(np.shape(dUx_dx))
    # tensor_displacement_list = map_elements_to_centraldiff(matrix, displacement_centroids, dUx_dx, dUy_dx, dUz_dx, dUx_dy, dUy_dy, dUz_dy, dUx_dz, dUy_dz, dUz_dz)
    # np.save('tensor_displacement_list.npy',tensor_displacement_list)
    
    os.chdir(r"C:\Users\yuvamk2\OneDrive - University of Illinois - Urbana\MS Thesis Files\UIUC MS Thesis Files\Codes\VFM")
    tensor_displacement_list=np.load('tensor_displacement_list.npy',allow_pickle=True)
    matrix=np.load('matrix.npy',allow_pickle=True)
    undeformed_centroids=np.load('undeformed_centroids.npy',allow_pickle=True)
    cube_size=np.load('cube_size.npy',allow_pickle=True)
    # phi_list=[]
    # for F in force_list:
    #     phi = calculate_deformation_gradient(tensor_displacement_list, matrix,E,v,undeformed_centroids,cube_size,F)
    #     phi_list.append(phi)
    # np.save('phi_list_vf2_2_force',phi_list)
    calculate_deformation_gradient(tensor_displacement_list, matrix,E,v,undeformed_centroids,cube_size,Force)
   
    # dUx_dx, dUy_dx, dUz_dx, dUx_dy, dUy_dy, dUz_dy, dUx_dz, dUy_dz, dUz_dz = central_differenciation(Ux_enlarged, Uy_enlarged, Uz_enlarged, cube_size)

    # tensor_displacement_list = map_elements_to_centraldiff(matrix, displacement_centroids, dUx_dx, dUy_dx, dUz_dx, dUx_dy, dUy_dy, dUz_dy, dUx_dz, dUy_dz, dUz_dz)
    # filename_tensor_displacement0 = 'tensor_displacement_list0.csv'
    # save_3d_matrix_to_csv(tensor_displacement_list, filename_tensor_displacement0)

    
    # # filename_dUx_dy_slice = 'dUx_dy.csv'

    # # save_3d_matrix_to_csv(dUx_dy, filename_dUx_dy_slice)

    # # Calculate deformation gradients for each element using the provided function
    # deformation_gradients = calculate_deformation_gradient(tensor_displacement_list, matrix,C1,C2,undeformed_centroids)
    # # filename_tensor_deformation0 = 'tensor_deformation_list0.csv'
    # # save_3d_matrix_to_csv(deformation_gradients, filename_tensor_deformation0)
    
    # # # Access and print the value of dUx_dx for the first element (index 0) in the matrix
    # # print("dUx_dx for the first element:")
    # # print(dUx_dx[0, 0, 0])  # Adjust the indices as needed to access other elements

    end = time.time()
    elapsed_time = end - initial
    # print(elapsed_time)

if __name__ == "__main__":
    main()













