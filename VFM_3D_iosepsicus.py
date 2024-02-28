
import numpy as np
import pandas as pd
import csv
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


decimal_places = 10
force = -700
index = 0
C1 = 2.5e+10
C2 = 0
L = 0.03 # Length in meters
H = 0.019 # Height in meters
D = 0.0023 # Depth in meters

df = pd.read_csv('iosipescu.csv')
df.head()
df=df.drop(columns=['ODB Name', 'Step', 'Frame', 'Section Name', 'Material Name', 'Section Point','X','Y','Z','   U-Magnitude'],axis=1)

nodes_displacement=df.loc[df['Part Instance Name'] == 'PART-1-1']
nodes_displacement=nodes_displacement.drop('Part Instance Name',axis=1)

print(nodes_displacement.head())
nodes_displacement = np.array(nodes_displacement)

def create_centroids(path, sheet1, sheet2, outfile, decimal_places):
    ## reading the coordinates and populating a 2d list
    data_coordinates = pd.read_excel(path, sheet_name=sheet1)
    df_coordinates = data_coordinates.iloc[:, :4]
    coordinates = []
    num_nodes = df_coordinates.shape[0]
    
    # Find the minimum x, y, and z values
    min_x = min(df_coordinates.iloc[:, 1])
    min_y = min(df_coordinates.iloc[:, 2])
    min_z = min(df_coordinates.iloc[:, 3])

    ## reading element to node connectivity and populating a 2d list
    data_connectivity = pd.read_excel(path, sheet_name=sheet2)
    df_connectivity = data_connectivity.iloc[:, :]
    connectivities = []
    num_connectivities = df_connectivity.shape[0]
    for i in range(num_connectivities):
        connectivities.append(df_connectivity.iloc[[i]].values.tolist()[0])

    # creates centroids based on node and element connectivity
    centroids = []
    for element in connectivities:
        nodal_coordinates = []
        for neighbors in element[1:]:
            nodal_coordinates.append(coordinates[neighbors-1][1:])
        x, y, z = list(np.mean(nodal_coordinates, axis=0))
        x = round(x, decimal_places)
        y = round(y, decimal_places)
        z = round(z, decimal_places)
        centroid = [element[0], x, y, z]
        centroids.append(centroid)

    with open(outfile, 'w', newline='') as file:
        writer = csv.writer(file)
        for centroid in centroids:
            writer.writerow(centroid)

    return centroids


def displacement(undeformed_centroids, deformed_centroids,decimal_places):
    displacement_centroids = []
    for i in range(len(deformed_centroids)):
        # Create a new list for each centroid to store the displacement
        displacement_centroid = [deformed_centroids[i][0]]

        # Calculate displacement in x, y, and z coordinates and round to 7 decimal places
        x_displacement = round(deformed_centroids[i][1] - undeformed_centroids[i][1], decimal_places)
        y_displacement = round(deformed_centroids[i][2] - undeformed_centroids[i][2], decimal_places)
        z_displacement = round(deformed_centroids[i][3] - undeformed_centroids[i][3], decimal_places)

        # Append the displacement values to the displacement_centroid list
        displacement_centroid.append(x_displacement)
        displacement_centroid.append(y_displacement)
        displacement_centroid.append(z_displacement)

        # Append the displacement_centroid to the displacement_centroids list
        displacement_centroids.append(displacement_centroid)
    return displacement_centroids


def calculate_cube_size(centroids):
    x_coordinates = [centroid[1] for centroid in centroids]
    y_coordinates = [centroid[2] for centroid in centroids]
    z_coordinates = [centroid[3] for centroid in centroids]

    #calculating the cube size
    cube_size = [0, 0, 0]
    x2 = max(x_coordinates)  
    x1 = 0.0                   
    
    for x_coordinate in x_coordinates:
        if x_coordinate < x2:
            x1 = max(x1, x_coordinate)

    y2 = max(y_coordinates)
    y1 = 0.0 
    
    for y_coordinate in y_coordinates:
        if y_coordinate < y2:
            y1 = max(y1, y_coordinate)

    z2 = max(z_coordinates)
    z1 = 0.0
    
    for z_coordinate in z_coordinates:
        if z_coordinate < z2:
            z1 = max(z1, z_coordinate)

    cube_size[0] = x2-x1
    cube_size[1] = y2-y1
    cube_size[2] = z2-z1

    cube_size[0] = round(cube_size[0], decimal_places)
    cube_size[1] = round(cube_size[1], decimal_places)
    cube_size[2] = round(cube_size[2], decimal_places)

    print(cube_size[0])
    print(cube_size[1])
    print(cube_size[2])

    return cube_size[0], cube_size[1], cube_size[2]


def map_elements_to_matrix(centroids,cube_size):

    
    # Calculate rows and columns for matrix to fill
    cols = int((max([centroid[1] for centroid in centroids]))/cube_size[0])+1
    rows = int((max([centroid[2] for centroid in centroids]))/cube_size[1])+1
    deps = int((max([centroid[3] for centroid in centroids]))/cube_size[2])+1
    
    # print(rows)
    # print(cols)
    # print(deps)

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
    tensor_displacement_list = np.zeros((40,60,5), dtype=object)

    for i in range(40):
        for j in range(60):
            for k in range(5):
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


def calculate_deformation_gradient(tensor_displacement_list, matrix,C1,C2,undeformed_centroids):
    deformation_gradients = tensor_displacement_list
    # Identity matrix (3x3)
    rows = len(tensor_displacement_list)
    cols = len(tensor_displacement_list[0])
    deps = len(tensor_displacement_list[0][0])
    I = np.eye(3)
    for i in range(rows):
        for j in range(cols):
           for k in range(deps):
                deformation_gradients[i][j][k] += I
    
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
    
    deformation_2d_list.sort()
    filename = 'deformation_gradients_2d.csv'
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in deformation_2d_list:
            writer.writerow(row)
    
    F = []
    C = []
    B = []
    E = []
    pi = []
    IVW_1 = []
    IVW_2 = []

    size = 12000
    deformation_2d_array = np.zeros((size, 9), dtype=float)
    pk1_2d_list = np.zeros((size, 9)) #change row number accordingly 
    e_2d_list = np.zeros((size, 9)) #change row number accordingly 
    c_2d_list = np.zeros((size, 9)) #change row number accordingly 
    for i in range(size):
        c_reshaped = np.zeros((3,3), dtype=float)
        c_reshaped[0,0] = deformation_2d_list[i][1]
        c_reshaped[0,1] = deformation_2d_list[i][2]
        c_reshaped[0,2] = deformation_2d_list[i][3]
        c_reshaped[1,0] = deformation_2d_list[i][4]
        c_reshaped[1,1] = deformation_2d_list[i][5]
        c_reshaped[1,2] = deformation_2d_list[i][6]
        c_reshaped[2,0] = deformation_2d_list[i][7]
        c_reshaped[2,1] = deformation_2d_list[i][8]
        c_reshaped[2,2] = deformation_2d_list[i][9]
        c = np.dot(c_reshaped.T, c_reshaped)
        b = np.dot(c_reshaped, c_reshaped.T)
        I1 = np.trace(c)
        derivative = C1 * np.eye(3) + C2 * (I1 * np.eye(3) - c.T)
        pk2 = 2*derivative
        e = 0.5 * (c - np.eye(3))
        c_reshaped_matrix = c_reshaped.reshape((3, 3))
        pk1 = np.dot(c_reshaped,pk2)

      # Calculate X1 and X2 for each centroid
        X1 = undeformed_centroids[i][1]
        X2 = undeformed_centroids[i][2]

        # First set of virtual fields and their derivatives
        u1_1 = 0
        u2_1 = -X1

        du1_dX1_1 = 0
        du1_dX2_1 = 0
        du2_dX1_1 = -1
        du2_dX2_1 = 0

        # Second set of virtual fields and their derivatives
        u1_2 = X1 * (L - X1) * X2
        u2_2 = (X1**3 / 3) - L * (X1**2 / 2)

        du1_dX1_2 = (L - 2*X1) * X2
        du1_dX2_2 = X1 * (L - X1)
        du2_dX1_2 = X1**2 - L * X1
        du2_dX2_2 = 0

        # Calculate internal virtual work for first set of virtual fields Π : ∂û/∂X 
        ivw_1 = (pk1[0, 0] * du1_dX1_1 + pk1[1, 0] * du2_dX1_1 + pk1[0, 1] * du1_dX2_1 + pk1[1, 1] * du2_dX2_1)*(0.0005*0.000475*0.00046)

        # Calculate internal virtual work for second set of virtual fields Π : ∂û/∂X 
        ivw_2 = (pk1[0, 0] * du1_dX1_2 + pk1[1, 0] * du2_dX1_2 + pk1[0, 1] * du1_dX2_2 + pk1[1, 1] * du2_dX2_2)*(0.0005*0.000475*0.00046)


        k = 0
        j = 0
        pk1_2d_list[i][0] = pk1[k][j]
        pk1_2d_list[i][1] = pk1[k][j+1]
        pk1_2d_list[i][2]  = pk1[k][j+2]
        pk1_2d_list[i][3]  = pk1[k+1][j]
        pk1_2d_list[i][4]  = pk1[k+1][j+1]
        pk1_2d_list[i][5]  = pk1[k+1][j+2]
        pk1_2d_list[i][6]  = pk1[k+2][j]
        pk1_2d_list[i][7]  = pk1[k+2][j+1]
        pk1_2d_list[i][8]  = pk1[k+2][j+2]
                 
        k = 0
        j = 0
        e_2d_list[i][0] = e[k][j]
        e_2d_list[i][1] = e[k][j+1]
        e_2d_list[i][2]  = e[k][j+2]
        e_2d_list[i][3]  = e[k+1][j]
        e_2d_list[i][4]  = e[k+1][j+1]
        e_2d_list[i][5]  = e[k+1][j+2]
        e_2d_list[i][6]  = e[k+2][j]
        e_2d_list[i][7]  = e[k+2][j+1]
        e_2d_list[i][8]  = e[k+2][j+2]         
                    
        k = 0
        j = 0
        c_2d_list[i][0] = c[k][j]
        c_2d_list[i][1] = c[k][j+1]
        c_2d_list[i][2]  = c[k][j+2]
        c_2d_list[i][3]  = c[k+1][j]
        c_2d_list[i][4]  = c[k+1][j+1]
        c_2d_list[i][5]  = c[k+1][j+2]
        c_2d_list[i][6]  = c[k+2][j]
        c_2d_list[i][7]  = c[k+2][j+1]
        c_2d_list[i][8]  = c[k+2][j+2]    
 
       

        
        C.append(c)
        B.append(b)
        E.append(e)
        pi.append(pk1)
        F.append(c_reshaped_matrix)
        IVW_1.append(ivw_1)
        IVW_2.append(ivw_2)
          
        
        
    total_IVW_1 = np.sum(IVW_1)
    total_IVW_2 = np.sum(IVW_2)

    print(total_IVW_1)
    print(total_IVW_2)
    # Calculate external virtual work for first set of virtual fields 

    evw_1 = (-2 * force * L)

    # Calculate external virtual work for second set of virtual fields 
    evw_2 = (- force * L**3 / 6)

    print(evw_1)
    print(evw_2)

    # Cost function
    phi = (total_IVW_1 - evw_1) ** 2 + (total_IVW_2 - evw_2) ** 2

    filenameB = 'B.csv'
    filenameC = 'C.csv'
    filenameE = 'E.csv'
    filenamepk1 = 'pk1.csv'
    filenameF = 'F.csv'
    
    
    print(phi)

    with open(filenameB, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in B:
            writer.writerow(row)

    with open(filenameC, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in C:
            writer.writerow(row)
    
    with open(filenameE, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in E:
            writer.writerow(row)

    with open(filenamepk1, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in pi:
            writer.writerow(row)

    with open(filenameF, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in F:
            writer.writerow(row)
    
    
    # with open(filenamepk1_2d, 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     for row in pk1_2d_list:
    #         writer.writerow(row) 

    np.savetxt('pk1_2d.csv', pk1_2d_list,  delimiter=',')
    np.savetxt('e_2d.csv', e_2d_list,      delimiter=',')
    np.savetxt('c_2d.csv', c_2d_list,      delimiter=',')
    return deformation_gradients
    


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
    path = 'Hexa.xlsx'
    undeformed_sheet = 'Initial'        # sheet of the coordinates
    connectivity_sheet = 'Node_Connectivity'    # sheet of element to node connectivities
    outfile1 = 'undeformed_centroids.csv'
    outfile2 = 'deformed_centroids.csv'
    deformed_sheet = 'step4'
    undeformed_centroids = create_centroids(path, undeformed_sheet, connectivity_sheet, outfile1,decimal_places)
    aaaaa
    deformed_centroids = create_centroids(path, deformed_sheet, connectivity_sheet, outfile2,decimal_places)
    displacement_centroids = displacement(undeformed_centroids, deformed_centroids,decimal_places)

    # Write displacement centroids to a CSV file
    displacement_file = 'displacement_centroids.csv'
    with open(displacement_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Element', 'X Displacement', 'Y Displacement', 'Z Displacement'])  # Write header
        for row in displacement_centroids:
            writer.writerow(row)

    cube_size = calculate_cube_size(undeformed_centroids)
    cube_size = [cube_size[0], cube_size[1], cube_size[2]]
    matrix = map_elements_to_matrix(undeformed_centroids,cube_size)

    filename = 'Element_matrix.csv'
    # Save the 2D matrix of the first depth (depth=0) to a CSV file
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in matrix[:, :, index]:
            writer.writerow(row)

    # Map displacements to matrices
    Ux, Uy, Uz = map_elements_to_displacement(matrix, displacement_centroids)

    # Save displacement matrices to CSV files

    filename_Ux = 'Ux.csv'
    save_3d_matrix_to_csv(Ux, filename_Ux)

    filename_Uy = 'Uy.csv'
    save_3d_matrix_to_csv(Uy, filename_Uy)

    filename_Uz = 'Uz.csv'
    save_3d_matrix_to_csv(Uz, filename_Uz)

    Ux_enlarged = increase_matrix_size(Ux)
    Uy_enlarged = increase_matrix_size(Uy)
    Uz_enlarged = increase_matrix_size(Uz)

    # Save the enlarged displacement matrices to CSV files
    filename_Ux = 'Ux_enlarged.csv'
    save_3d_matrix_enalrged_to_csv(Ux_enlarged, filename_Ux)

    filename_Uy = 'Uy_enlarged.csv'
    save_3d_matrix_enalrged_to_csv(Uy_enlarged, filename_Uy)

    filename_Uz = 'Uz_enlarged.csv'
    save_3d_matrix_enalrged_to_csv(Uz_enlarged, filename_Uz)

    dUx_dx, dUy_dx, dUz_dx, dUx_dy, dUy_dy, dUz_dy, dUx_dz, dUy_dz, dUz_dz = central_differenciation(Ux_enlarged, Uy_enlarged, Uz_enlarged, cube_size)

    tensor_displacement_list = map_elements_to_centraldiff(matrix, displacement_centroids, dUx_dx, dUy_dx, dUz_dx, dUx_dy, dUy_dy, dUz_dy, dUx_dz, dUy_dz, dUz_dz)
    filename_tensor_displacement0 = 'tensor_displacement_list0.csv'
    save_3d_matrix_to_csv(tensor_displacement_list, filename_tensor_displacement0)

    
    # filename_dUx_dy_slice = 'dUx_dy.csv'

    # save_3d_matrix_to_csv(dUx_dy, filename_dUx_dy_slice)

    # Calculate deformation gradients for each element using the provided function
    deformation_gradients = calculate_deformation_gradient(tensor_displacement_list, matrix,C1,C2,undeformed_centroids)
    # filename_tensor_deformation0 = 'tensor_deformation_list0.csv'
    # save_3d_matrix_to_csv(deformation_gradients, filename_tensor_deformation0)
    
    # # Access and print the value of dUx_dx for the first element (index 0) in the matrix
    # print("dUx_dx for the first element:")
    # print(dUx_dx[0, 0, 0])  # Adjust the indices as needed to access other elements


if __name__ == "__main__":
    main()













