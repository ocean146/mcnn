def get_mse_mae(outputs_list,targets_list):
    assert len(outputs_list) > 0 and len(targets_list)==len(outputs_list)
    mse=0.0
    mae=0.0
    for i in range(len(outputs_list)):
        mae += abs(outputs_list[i] - targets_list[i])
        mse += (outputs_list[i] - targets_list[i])**2
    mae /= len(outputs_list)
    mse /= len(outputs_list)
    return mse,mae



# test
if __name__ == '__main__':
    a=[0,1,2,3]
    b=[0,0,0,0]
    mse,mae = get_mse_mae(a,b)
    print(F"{mse=},{mae=}")
