from math import e
import torch
import torch.nn.functional as F
import numpy as np


def cuda_dist(x, y):
    #dist = 1 - q_features_norm @ g_features_norm.T
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    dist = torch.sum(x ** 2, 1).unsqueeze(1) + torch.sum(y ** 2, 1).unsqueeze(
        1).transpose(0, 1) - 2 * torch.matmul(x, y.transpose(0, 1))
    dist = torch.sqrt(F.relu(dist))
    return dist


def evaluation_old(data, config):
    dataset = config['dataset'].split('-')[0]
    print(f"Dataset: {dataset}")
    # feature shape = (5485, 15872)
    # view = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162',...], length = 5485
    # seq_type = ['bg-01', 'bg-01', 'bg-01', 'bg-01', 'bg-01', 'bg-01',...], length = 5485
    # label = ['075', '075', '075', '075', '075', '075', '075',...], length = 5485
    feature, view, seq_type, label = data

    print("data: ", data)
    print(f"feature shape: {feature.shape}")
    print(f"view shape: {len(view)}")
    print(f"label shape: {len(label)}")
    
    used_data = {}
    #info = []
    #features = []
    for i in range(len(label)):
        if (view[i] == '045' or view[i] == '135') and label[i] != '09':
            #info.append([label[i], seq_type[i], view[i]])
            #features.append(feature[i])
            k = (label[i], seq_type[i], view[i])
            used_data[k] = feature[i]
    
    label = np.array(label)
    print(f"Labels: {label}")
    #view_list = list(set(view))
    #view_list.sort()
    view_list = ['045', '135']
    #view_list = ['045']
    #view_list = ['135']
    print(f"view_list: {view_list}")
    view_num = len(view_list)
    sample_num = len(feature)

    print(view_num)

    probe_seq_dict = {'CASIA': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
                      'OUMVLP': [['00']],
                      'Asilla': [['ok-13', 'ok-14', 'ok-15', 'ok-16', 'ok-17', 'ok-18',
                                  'ok-19', 'ok-20', 'ok-21']]}
    gallery_seq_dict = {'CASIA': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
                        'OUMVLP': [['01']],
                        'Asilla': [['ok-01', 'ok-02', 'ok-03', 'ok-04',
                                    'ok-05', 'ok-06', 'ok-07', 'ok-08', 'ok-09',
                                    'ok-10', 'ok-11', 'ok-12']]}

    # calculate mAP here

    print(f"probe_seq_dict {probe_seq_dict[dataset]}")
    num_rank = 1
    acc = np.zeros([len(probe_seq_dict[dataset]),
                   view_num, view_num, num_rank])
    print(f"From eval: {acc.shape}")
    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        for gallery_seq in gallery_seq_dict[dataset]:
            for (v1, probe_view) in enumerate(view_list):
                for (v2, gallery_view) in enumerate(view_list):
                    # gallery_seq = ['nm-01', 'nm-02', 'nm-03', 'nm-04']
                    # get gallery from particular view = set A
                    gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(
                        view, [gallery_view])

                    # get all the features of set A, shape = (200, 15872)
                    gallery_x = feature[gseq_mask, :]
                    # get all the labels of set A, length = 200
                    gallery_y = label[gseq_mask]

                    # get probe from particular view = set B
                    pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                        view, [probe_view])
                    # get all the features of set B, shape = (100, 15872)
                    probe_x = feature[pseq_mask, :]
                    # get all the labels of set B, length = 100
                    probe_y = label[pseq_mask]

                    # distance between 2 set of features (feature of set A and set B), size (100, 200)
                    dist = cuda_dist(probe_x, gallery_x)
                    # sort the distance in ascending order, get the original indices after sorted, shape = (100, 200)
                    #---------------------------------------------------#
                    idx = dist.sort(1)[1].cpu().numpy()

                    # idx[:, 0:num_rank] = [[  1   3   0   2  47] [  1   3   2   0  25]...], shape = (100, num_rank)
                    # gallery_y[idx[:, 0:num_rank]] = [['075' '075' '075' '075' '086']['075' '075' '075' '075' '081']...], shape = (100, num_rank)
                    # np.reshape(probe_y, [-1, 1]).shape = (100, 1)
                    # (np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]]).shape = (100, num_rank)

                    # print((np.round(
                    #     np.sum(np.sum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                    #            0) * 100 / dist.shape[0], 2)).shape)

                    # acc[p, v1, v2, :] = np.round(
                    #     np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                    #            0) * 100 / dist.shape[0], 2)

                    acc[p, v1, v2, :] = np.round(
                        np.sum(np.sum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]],
                                      1, keepdims=True) > 0,
                               0) * 100 / dist.shape[0], 2)

    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        print(f"p: {p}, probo_seq: {probe_seq}")
        for gallery_seq in gallery_seq_dict[dataset]:
            print(f"Gallery_seq: {gallery_seq}")
            for (v1, probe_view) in enumerate(view_list):
                for (v2, gallery_view) in enumerate(view_list):
                    print(f"gallery_view: {gallery_view}")
                    print(f"probe_view: {probe_view}")
                    # gallery_seq = ['nm-01', 'nm-02', 'nm-03', 'nm-04']
                    # get gallery from particular view = set A
                    gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(
                        view, [gallery_view])

                    # get all the features of set A, shape = (200, 15872)
                    gallery_x = feature[gseq_mask, :]
                    print(f"gallery_x: {gallery_x}")
                    # get all the labels of set A, length = 200
                    gallery_y = label[gseq_mask]
                    print(f"gallery_y: {gallery_y}")

                    # get probe from particular view = set B
                    pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                        view, [probe_view])
                    # get all the features of set B, shape = (100, 15872)
                    probe_x = feature[pseq_mask, :]
                    # get all the labels of set B, length = 100
                    probe_y = label[pseq_mask]
                    print(f"probe_y: {probe_y}")

                    correct = np.zeros((3, 11, 11))
                    total = np.zeros((3, 11, 11))

                    false_cl = 0
                    list_false_cl = []
                    list_mistake = []
                    gallery_angle = 180
                    probe_num = 0

                    for probe_id in probe_y:
                        # distance between 2 set of features (feature of set A and set B), size (100, 200)
                        dist = cuda_dist(probe_x, gallery_x)
                        print(f"dist shape: {dist.shape}")
                        print(f"dist: {dist}")

                        idx = dist.sort(1)[1].cpu().numpy()
                        print(f"idx: {idx}")
                        min_pos = np.argmax(dist.cpu().numpy(), axis=0)
                        print(f"min_pos: {min_pos}")
                        min_target = gallery_y[min_pos]

                        print(f"Testing min_target: {min_target}")

                        if min_target[0] == probe_id:
                            #correct[probe_num, gallery_view, probe_view] += 1.0
                             print('true', min_target, (probe_id, probe_view))
                             print('=====')
                        # total[probe_num, gallery_pos, probe_pos] += 1

                        else:
                            false_cl += 1
                            print('false', min_target, (probe_id,probe_view))
                            print('=====', false_cl)
                            
                            list_false_cl.append(((probe_id,probe_view, probe_seq),min_target))
                            list_mistake.append((probe_id,min_target))
                        # print(list_false)
    list_false_cl = np.asarray(list_false_cl)
    list_mistake = np.asarray(list_mistake)
    print(f"List Mistake: {list_mistake}")
    print(f"List False: {list_false_cl}")

    return acc

def evaluation(data, config):
    dataset = config['dataset'].split('-')[0]
    print(f"Dataset: {dataset}")
    # feature shape = (5485, 15872)
    # view = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162',...], length = 5485
    # seq_type = ['bg-01', 'bg-01', 'bg-01', 'bg-01', 'bg-01', 'bg-01',...], length = 5485
    # label = ['075', '075', '075', '075', '075', '075', '075',...], length = 5485
    feature, view, seq_type, label = data

    print("data: ", data)
    print(f"feature shape: {feature.shape}")
    print(f"view shape: {len(view)}")
    print(f"label shape: {len(label)}")
    
    used_data = {}
    #info = []
    #features = []

    #for only angles 45 and 135
    # for i in range(len(label)):
    #     if (view[i] == '045' or view[i] == '135'):
    #         #info.append([label[i], seq_type[i], view[i]])
    #         #features.append(feature[i])
    #         k = (label[i], seq_type[i], view[i])
    #         used_data[k] = feature[i]

    #all angles in dataset
    # for i in range(len(label)):
    #     #info.append([label[i], seq_type[i], view[i]])
    #     #features.append(feature[i])
    #     k = (label[i], seq_type[i], view[i])
    #     used_data[k] = feature[i]

    #only angle 45 in dataset
    # for i in range(len(label)):
    #     if view[i] == '045':
    #         #info.append([label[i], seq_type[i], view[i]])
    #         #features.append(feature[i])
    #         k = (label[i], seq_type[i], view[i])
    #         used_data[k] = feature[i]

    #only angle 135 in dataset
    for i in range(len(label)):
        if view[i] == '135':
            #info.append([label[i], seq_type[i], view[i]])
            #features.append(feature[i])
            k = (label[i], seq_type[i], view[i])
            used_data[k] = feature[i]
    

    #info = np.asarray(info)
    #features = np.asarray(features)

    #np.save('/home/minhdoan/Documents/projects/jupyter-notebook/asilla/mct-project/GaitSet-TL/work/info.npy',
    #         info)
    #np.save('/home/minhdoan/Documents/projects/jupyter-notebook/asilla/mct-project/GaitSet-TL/work/features.npy',
    #         features)


    print(f"length of used_data: {len(used_data)}")
    print(f"Used_data: {used_data}")

    gallery = []
    probe = []
    idx_gallery = []

    num_id = 4
    for idx in range(1, 11):
        idx_gallery.append(num_id)
        num_id += 2

    num_id = 5
    for idx in range(1, 11):
        idx_gallery.append(num_id)
        num_id += 2

    print(idx_gallery)
    
    for (k, v) in used_data.items():
        angle = int(k[1].split('-')[1])
        #print(angle)
        if (angle in idx_gallery):
            gallery.append((k, v))
        else:
            # if (int(k[3]) == 135):
            probe.append((k, v))

    print('len gallery', len(gallery))
    print('len probe', len(probe))

    g_features = []
    g_pids = []
    g_camids = []

    q_features = []
    q_pids = []
    q_camids = []

    for g in gallery:
        targer, embedding = g
        if True:
            # print('---targer---', targer)
            subject_id, _, gallery_angle = targer  # ids and camids of gallery
            g_features.append(embedding)
            g_pids.append(subject_id)
            g_camids.append(gallery_angle)

    for p in probe:
        target, embedding = p
        if True:
            subject_id, _, probe_angle = target  # ids and camids of query
            q_features.append(embedding)
            q_pids.append(subject_id)
            q_camids.append(probe_angle)

    q_features = np.array(q_features)
    g_features = np.array(g_features)
    # print(q_features.shape, g_features.shape)

    q_features_norm = q_features / \
        np.linalg.norm(q_features, axis=1)[:, np.newaxis]
    g_features_norm = g_features / \
        np.linalg.norm(g_features, axis=1)[:, np.newaxis]

    print('-------------')
    print(q_features_norm.shape)
    print(g_features_norm.shape)
    print('-------------')

    dismat = 1 - q_features_norm @ g_features_norm.T
    print('dismat', dismat)


    print("\n---------------------------------------------------------------------------------\n")

    gallery2 = {k: v for (k, v) in gallery}
    probe2 = {k: v for (k, v) in probe}
    print(type(gallery2), type(probe2))

    correct = np.zeros((3, 11, 11))
    total = np.zeros((3, 11, 11))

    false_cl = 0
    list_false_cl = []
    list_mistake = []
    gallery_angle = 180
    probe_num = 0

    for num in [gallery2]:
        gallery_targets = list(gallery2.keys())
        gallery_pos = int(gallery_angle / 18)

    for p in [probe2]:
        for (target, embedding) in p.items():
            print(f"Embedding: {embedding}")
            subject_id, _, probe_angle = target
            probe_pos = int(probe_angle) / 18

            embedding_norm = embedding / \
                np.linalg.norm(embedding)

            # if(int(probe_angle) == 135):
            print('-------------')
            print((embedding.T).shape)
            print(g_features.shape)
            print('-------------')
            #distance = np.linalg.norm(g_features - embedding, ord=2, axis=1)
            #distance = g_features@embedding.T #np.linalg.norm(g_features - embedding, ord=2, axis=1)
            distance = np.linalg.norm(g_features_norm - embedding_norm, ord=2, axis=1)
            min_pos = np.argmin(distance)
            min_target = gallery_targets[int(min_pos)]
            """print(distance.shape)
            min_pos = np.argmax(distance)
            min_target = gallery_targets[int(min_pos)]"""
            # print(min_target)

            if min_target[0] == subject_id:
                #correct[probe_num, gallery_pos, probe_pos] += 1
                print('true', min_target, target)
                print('=====')
            # total[probe_num, gallery_pos, probe_pos] += 1

            else:
                false_cl += 1
                print('false', min_target, target)
                print('=====', false_cl)
                for pos in np.argsort(distance)[-1:-20:-1]:
                    print('---- pos {} {} dist {}'.format(pos, gallery_targets[int(pos)], distance[pos]))

                for pos in np.argsort(distance)[:-20]:
                                    print('---- pos {} {} dist {} -neg'.format(pos, gallery_targets[int(pos)], distance[pos]))

            list_false_cl.append(target)
            list_mistake.append(min_target)
    # print(list_false)
    list_false_cl = np.asarray(list_false_cl)
    list_mistake = np.asarray(list_mistake)
    print(list_mistake.shape)

    q_pids = np.array(q_pids)
    g_pids = np.array(g_pids)
    # print(dismat.shape, q_pids.shape, g_pids.shape)
    cmc, mAP = evaluate_py_new(dismat, q_pids, g_pids, q_camids, g_camids)

    print('** Results **')
    print('mAP: {:.1%}'.format(mAP))
    print('CMC curve')
    for r in [1, 5, 10, 20]:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))

    return cmc[0]


def evaluate_py_new(distmat, q_pids, g_pids, q_camids, g_camids):
    CMC = torch.IntTensor(len(g_pids)).zero_()
    ap = 0.0
    # print(query_label)
    #print(distmat.shape, q_pids.shape, g_pids.shape, q_camids.shape, g_camids.shape)
    #print(q_pids[:10], g_pids[:10])
    for i in range(len(q_pids)):
        # print(i)
        ap_tmp, CMC_tmp = evaluate2(
            distmat[i], q_pids[i], q_camids[i], g_pids, g_camids)
        if CMC_tmp[0] == -1:
            continue
        # print(CMC.shape, CMC_tmp.shape, distmat[i].shaoem q_pids.shape, g_pids.shape)
        CMC += CMC_tmp
        ap += ap_tmp
        # raise
        #print(i, CMC_tmp[0])
        #raise SystemExit

    CMC = CMC.float()
    CMC = CMC/len(q_pids)  # average CMC
    ap /= len(q_pids)
    print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap))
    return CMC.numpy(), ap


def evaluate2(score, ql, qc, gl, gc):
    #query = qf
    #score = np.dot(gf,query)
    # predict index
    index = np.argsort(score)  # from small to large

    #index = index
    #index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)
    #print(gl, ql, query_index)
    camera_index = np.argwhere(gc == qc)

    #good_index = query_index
    # print(good_index.shape)
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    # print('index', index)
    # print('good', good_index)
    # print('query', query_index)
    # print('cam', camera_index)

#     raise ValueError()
    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)  # .flatten())
    #print(good_index, junk_index)
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:   # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)

        if rows_good[i] != 0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision = 1.0

        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc


def mAP(data, config):
    dataset = config['dataset'].split('-')[0]
    feature, view, seq_type, label = data
    print(f"from mAP: {data}")
    label = np.array(label)
    view = np.array(view)
    view_list = list(set(view))
    view_list.sort()
    view_num = len(view_list)
    sample_num = len(feature)

    probe_seq_dict = {'CASIA': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
                      'OUMVLP': [['00']],
                      'Asilla': [['ok-13', 'ok-14', 'ok-15', 'ok-16', 'ok-17', 'ok-18',
                                  'ok-19', 'ok-20', 'ok-21']]}
    gallery_seq_dict = {'CASIA': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
                        'OUMVLP': [['01']],
                        'Asilla': [['ok-01', 'ok-02', 'ok-03', 'ok-04',
                                    'ok-05', 'ok-06', 'ok-07', 'ok-08', 'ok-09',
                                    'ok-10', 'ok-11', 'ok-12']]}

    # calculate mAP here

    g_features = []
    g_pids = []
    g_camids = []

    q_features = []
    q_pids = []
    q_camids = []

    dismat = None

    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        for gallery_seq in gallery_seq_dict[dataset]:
            for (v1, probe_view) in enumerate(view_list):
                distmat_temp = None
                for (v2, gallery_view) in enumerate(view_list):
                    gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(
                        view, [gallery_view])
                    g_features = feature[gseq_mask, :]

                    pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                        view, [probe_view])
                    q_features = feature[pseq_mask, :]

                    # q_features_norm = q_features / np.linalg.norm(q_features, axis=1)[:,np.newaxis]
                    # g_features_norm = g_features / np.linalg.norm(g_features, axis=1)[:,np.newaxis]
                    # dist = 1 - q_features_norm @ g_features_norm.T
                    dist = cuda_dist(q_features, g_features).cpu().numpy()

                    if distmat_temp is None:
                        distmat_temp = dist
                    else:
                        distmat_temp = np.concatenate(
                            (distmat_temp, dist), axis=1)

                if dismat is None:
                    dismat = distmat_temp
                else:
                    dismat = np.concatenate((dismat, distmat_temp), axis=0)

    for gallery_seq in gallery_seq_dict[dataset]:
        for (v2, gallery_view) in enumerate(view_list):
            gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(
                view, [gallery_view])
            g_pids.extend(label[gseq_mask])
            g_camids.extend(view[gseq_mask])

    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        for (v1, probe_view) in enumerate(view_list):
            pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                view, [probe_view])
            q_pids.extend(label[pseq_mask])
            q_camids.extend(view[pseq_mask])

    q_pids = np.array(q_pids)
    g_pids = np.array(g_pids)

    cmc, mAP = evaluate_py_new(dismat, q_pids, g_pids, q_camids, g_camids)

    print('===mAP: {:.1%}'.format(mAP))

    for r in [1,5,10,20]:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))

    return cmc[0]
