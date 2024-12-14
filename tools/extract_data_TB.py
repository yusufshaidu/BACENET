def get_stochastic_RMSE(path):
    events = EventAccumulator(path,  size_guidance= {'tensors': 1}).Reload()

    keys = ['RMSE_F', 'RMSE/atom', 'Rs_rad', 'Rs_ang', 'width', 'width_ang', 'zeta', 'thetas']
    my_data = {}
    for key in keys:
        my_data[key] = [[], []]
        for item in events.Tags()['tensors']:
            if key == item.split()[-1]:
                break
                print(item)

        d = events.Tensors(item)

        for i in range(len(d)):

            my_data[key][0].append(pandas.DataFrame(d)['step'][i])
            my_data[key][1].append(tf.make_ndarray(d[i].tensor_proto))

    return my_data
