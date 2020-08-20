import torch.optim as optim
from isonet.utils.config import C


def construct_optim(net1, net2, num_gpus):
    # channel-shared parameters.
    shared_params = []
    # Non-batchnorm parameters.
    other_params = []
    for name, p in net1.named_parameters():
        if 'shared' in name:
            shared_params.append(p)
        else:
            other_params.append(p)

    for name, p in net2.named_parameters():
        if 'shared' in name:
            shared_params.append(p)
        else:
            other_params.append(p)

    optim_params = [
        {
            'params': other_params,
            'weight_decay': C.SOLVER.WEIGHT_DECAY,
        },
        {
            'params': shared_params,
            'lr': C.SOLVER.BASE_LR / 10,
            'scaling': 0.1,
        }
    ]
    # Check all parameters will be passed into optimizer.
    assert len(list(net1.parameters())) + len(list(net2.parameters())) == len(other_params) + len(shared_params), \
        f'parameter size does not match: ' \
        f'{len(other_params)} + {len(shared_params)} != ' \
        f'{len(list(net1.parameters()))} + {len(list(net2.parameters()))}'

    return optim.SGD(
        optim_params,
        lr=C.SOLVER.BASE_LR * num_gpus,
        momentum=C.SOLVER.MOMENTUM,
        weight_decay=C.SOLVER.WEIGHT_DECAY,
        dampening=C.SOLVER.DAMPENING,
        nesterov=C.SOLVER.NESTEROV
    )
