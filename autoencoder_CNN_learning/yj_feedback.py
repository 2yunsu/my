import random

import torch
class CNN:
    pass
class AE :
    pass
# https://velog.io/@ohado/Seed-고정
if __name__ == '__main__':
    random_seed = 0
    torch.manual_seed(random_seed)  # torch
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True  # cudnn
    torch.backends.cudnn.benchmark = False  # cudnn
    np.random.seed(random_seed)  # numpy
    random.seed(random_seed)  # random
    cnn = CNN()
    ae = AE()
    epochs = 100
    threshold  = 0.1 # 매우 중요.!
    cnt_over_thres = 0
    cnt_under_thres = 0
    # train phase
    for j in range(epochs):
        for i in range(train_loader):
            input_data, train_label = train_batch()
            out = ae(input_data)
            loss = out - input_data
            loss.backward()
            optimizer.step()
            threshold = loss.item() 에서 상위 5%의 값
            if loss.item() <= threshold:
                output = cnn(input_data)
                label = output
                label = label.argmax()
                loss_cnn = cross_entropy(label, output)
                loss_cnn.backward()
                optimizer_cnn.step()
                cnt_under_thres += 1
            else: # 처음 받아들이는 경우
                output = cnn(input_data)
                label = input_data_label
                loss_cnn = cross_entropy(input_data_label, output)
                loss_cnn.backward()
                optimizer_cnn.step()
                cnt_over_thres += 1

    # test phase
    test_data = get_test_load()
    test_output = cnn(test_data)
    accur = accuracy(test_output, test_data_label)
    print('accur', accur)