import numpy as np
import random
import math
import matplotlib.pyplot as plt
import copy



# Location initialization
def node_initialization(lattice):
    '''
    :param lattice: location is within [-L/2, L/2]
    :return:
    '''

    # Location initialization
    y_pos = random.uniform(-lattice/2, lattice/2)
    x_pos = random.uniform(-lattice/2, lattice/2)
    # Popularity initialization
    b_popularity = random.uniform(1,2)
    # Direction initialization
    theta = random.uniform(0,2*math.pi)

    return x_pos, y_pos, b_popularity, theta


def graph_generation(locations, b_population,epsilo,d):
    '''

    :param locations: #[N,2]
    :param b_population: #[N]
    :return: #[N,N]
    '''
    x_pos = copy.deepcopy(locations[:, 0])
    y_pos = copy.deepcopy(locations[:, 1])
    x_diff_matrix = np.concatenate([np.asarray(x_pos).reshape(1, -1) for _ in range(len(x_pos))],
                                   axis=0) - np.concatenate(
        [np.asarray(x_pos).reshape(-1, 1) for _ in range(len(x_pos))], axis=1)  # [N,N]
    x_diff_matrix = x_diff_matrix ** 2

    y_diff_matrix = np.concatenate([np.asarray(y_pos).reshape(1, -1) for _ in range(len(y_pos))],
                                   axis=0) - np.concatenate(
        [np.asarray(y_pos).reshape(-1, 1) for _ in range(len(y_pos))], axis=1)  # [N,N]
    y_diff_matrix = y_diff_matrix ** 2
    diff_matrix = x_diff_matrix + y_diff_matrix

    b_matrix = np.concatenate([np.asarray(b_population).reshape(1, -1) for _ in range(len(b_population))],
                              axis=0) * np.concatenate(
        [np.asarray(b_population).reshape(-1, 1) for _ in range(len(b_population))], axis=1)  # [N,N]

    edge_matrix = np.exp(-diff_matrix / b_matrix / epsilo / epsilo)

    # Convert to [0,1] Matrix
    edge_matrix = np.where(edge_matrix>d,1,0)
    print(np.sum(edge_matrix))

    return edge_matrix



def opinion_migration(locations,thetas,graph,v,noise_sigma):
    '''
    :param locations: [N,2]
    :param thetas: [N]
    :param graph: [N,N]
    :param v: absolute velocity
    :return:
    '''

    # sample new thetas: [N]
    locations_tmp = copy.deepcopy(locations)
    thetas_expanded = np.concatenate([thetas.reshape(1, -1) for _ in range(len(thetas))],
                                   axis=0) #[N,N]
    thetas_avg =np.sum(graph*thetas_expanded,axis=1) #[N]
    num_neighbors = np.sum(graph,axis=1) #[N]
    thetas_new_mean = thetas_avg/num_neighbors #[N]

    thetas_new = np.random.normal(thetas_new_mean,noise_sigma) #[N]

    locations_new = np.zeros_like(locations)

    #Update locations

    locations_tmp[:,0] +=v*np.cos(thetas_new)
    locations_tmp[:,1] += v*np.sin(thetas_new)

    #return locations_new,thetas_new
    return locations_tmp, thetas_new



def draw_locations(locations,T):
    x = locations[:,0]
    y = locations[:,1]

    plt.scatter(x,y,c = "blue")
    plt.title("t=%d" % T)
    plt.show()





if __name__ == "__main__":
    # Hyperparatemer initialization
    T = 400
    random.seed(10)
    L = 5
    N = 80
    v = 0.03
    viz = 50

    noise_sigma = 0.2
    epsilo = 0.5
    d = np.exp(-0.4)


    # Initial Position Generation
    b_popularity_list= []
    locations = np.zeros((N, 2))
    thetas= []
    for i in range(N):
        x_pos, y_pos, b_popularity, theta = node_initialization(L)
        b_popularity_list.append(b_popularity)
        locations[i, 0] = x_pos
        locations[i, 1] = y_pos
        thetas.append(theta)

    b_popularity_list = np.asarray(b_popularity_list)
    thetas = np.asarray(thetas)

    # Generatiobn Process
    thetas_list = []   #[T+1]
    locations_list = [] #[T+1]
    graphs_list = [] #[T]

    #Initial position
    locations_input = copy.deepcopy(locations)
    thetas_input = thetas
    thetas_list.append(thetas)
    locations_list.append(locations)

    draw_locations(locations,0)

    for i in range(1,T):
        ## Step1: Graph Generation
        graph = graph_generation(locations_input,b_popularity_list,epsilo,d)
        graphs_list.append(graph)
        ## Step2: Opinion Migration
        locations_input,thetas_input = opinion_migration(locations_input,thetas_input,graph,v,noise_sigma)
        thetas_list.append(thetas_input)
        locations_list.append(locations_input)
        if i%viz == 0:
            draw_locations(locations_input,i)




    # save graphs
    locations_list = np.asarray(locations_list)
    graphs_list = np.asarray(graphs_list)

    np.save("locations_d_"+str(d) + "_noise_" + str(noise_sigma)+".npy",locations_list)
    np.save("graphs_d_"+str(d) + "_noise_" + str(noise_sigma)+".npy",graphs_list)
    np.save("popularity_d_"+str(d) + "_noise_" + str(noise_sigma)+".npy",b_popularity_list)










