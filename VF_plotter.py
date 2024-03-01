import numpy as np
import matplotlib.pyplot as plt
# Arbitrary values
l = 1*1e-3
w = 1*1e-3
h = 3
C = 1 # Constant magntiude value
k1=2
k2 = 15
a_0 = 2.7 # Contact radius * something
z_a_0=0.5


n = 1000 # Data points for plot
def U_star(r, z, a_0, C, h, k1,k2):
    C = 1 # Constant magntiude value
    k1=2
    k2 = 15
    a_0 = 2.7 # Contact radius * something
    z_a_0=0.5
    t = z / h
    return C*(1-(1/(1+np.exp(-k1*(r-a_0)))))
    # return C*(1-(1/(1+np.exp(-k1*(r-a_0)))))*(1/(1+np.exp(-k2*(t-z_a_0))))

def U_star_2(r,z,a_0, C, h, k1,k2):
    t=z/h
    Rs=5e-4
    delta=1e-4
    Rc=0.3
    a=0.2*1e-3
    k1=5e4
    C=-1e4
    return C*(delta-Rs+np.sqrt(np.abs(Rs**2 - r**2)))*(1-1/(1+np.exp(-k1*(r-a))))*t
    # return (1-1/(1+np.exp(-k1*(r-a_0))))

def plot_slices(plane, values, l, w, h, a_0, C):
    if plane == 'XY':
        x = np.linspace(-l / 2, l / 2, n)
        y = np.linspace(-w / 2, w / 2, n)
        X, Y = np.meshgrid(x, y)
        Z_values = values
        xlabel, ylabel, title = 'X1', 'X2', 'h'
    elif plane == 'XZ':
        z = np.linspace(0, h, n)
        y = np.linspace(-w / 2, w / 2, n)
        Z, Y = np.meshgrid(z, y)
        X_values = values
        xlabel, ylabel, title = 'X3', 'X2', 'l'
    global_min = np.inf
    global_max = -np.inf
    for value in values:
        if plane == 'XY':
            Z = np.full_like(X, value)
            U_star_values = U_star_2(np.sqrt(X**2 + Y**2), Z, a_0, C, h, k1,k2)
        elif plane == 'XZ':
            X = np.full_like(Z, value)
            U_star_values = U_star_2(np.sqrt(X**2 + Y**2), Z, a_0, C, h, k1,k2)
        global_min = min(global_min, U_star_values.min())
        global_max = max(global_max, U_star_values.max())
    # Plotting slices
    fig, axs = plt.subplots(1, len(values), figsize=(15, 4 if plane == 'XY' else 3))
    vmin = global_min
    vmax = global_max
    for i, value in enumerate(values):
        if plane == 'XY':
            Z = np.full_like(X, value)
            U_star_values = U_star_2(np.sqrt(X**2 + Y**2), Z, a_0, C, h, k1,k2)
        elif plane == 'XZ':
            X = np.full_like(Z, value)
            U_star_values = U_star_2(np.sqrt(X**2 + Y**2), Z, a_0, C, h, k1,k2)
        # vmin and vmax for all subplots
        if plane == 'XY':
            c = axs[i].contourf(X, Y, U_star_values, cmap='viridis', levels=10, vmin=vmin, vmax=vmax)
            # plt.axvline(x=0.3)
        elif plane == 'XZ':
            c = axs[i].contourf(Z, Y, U_star_values, cmap='viridis', levels=10, vmin=vmin, vmax=vmax)
        axs[i].set_xlabel(xlabel)
        axs[i].set_ylabel(ylabel)
        axs[i].set_title(f'{title} = {value:.2f}')
    # Create a common colorbar
    fig.colorbar(c, ax=axs, orientation='vertical', label='U* Values')
    plt.suptitle(f'Slices of $U^*$ in the {plane}-plane')
    plt.show()
z_values = [h,h/2,0]
x_values = [l/2, l/4, l/8, 0]
# plot_slices('XY', z_values, l, w, h, a_0, C)
# plot_slices('XZ', x_values, l, w, h, a_0, C)


U_star_values=[]
r=np.linspace(0,1,1001)*1E-3
# r=0.3
# z=h*np.linspace(0,1,101)
z=3e-3
U_star_values.append(U_star_2(r, z, a_0, C, h, k1,k2))



U_star_values=np.array(U_star_values)
U_star_values=U_star_values.flatten()
# print(np.shape(U_star_values))
plt.plot(r,U_star_values)
plt.axvline(x=0.3*1E-3)
plt.show()
print(U_star_values)


# U_star_values=[]
# # r=np.linspace(0,7.5,50)
# r=0.3
# z=h*np.linspace(0,1,101)
# # z=h
# U_star_values.append(U_star(r, z, a_0, C, h,k1,k2))
# U_star_values=np.array(U_star_values)
# U_star_values=U_star_values.flatten()
# # print(np.shape(U_star_values))
# plt.plot(z,U_star_values)
# # plt.axvline(x=0.3)
# plt.axvline(x=2.9)
# plt.show()

