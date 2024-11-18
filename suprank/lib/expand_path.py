import os


def expand_path(pth: str) -> str:
    pth = os.path.expandvars(pth)
    pth = os.path.expanduser(pth)

    if ('/local/DEEPLEARNING' in pth) and (not os.path.exists('/local/DEEPLEARNING')):
        if os.path.exists('/local/SSD_DEEPLEARNING_1'):
            pth = pth.replace('/local/DEEPLEARNING', '/local/SSD_DEEPLEARNING_1')

            if not os.path.exists(pth):
                pth = pth.replace('/local/SSD_DEEPLEARNING_1', '/local/SSD_DEEPLEARNING_2')

        elif os.getenv('SCRATCH'):
            if '/local/DEEPLEARNING' in pth:
                pth = pth.replace('/local/DEEPLEARNING/image_retrieval', os.path.expandvars('$SCRATCH'))
            elif '/local/SSD_DEEPLEARNING_1' in pth:
                pth = pth.replace('/local/SSD_DEEPLEARNING_1/image_retrieval', os.path.expandvars('$SCRATCH'))
            elif '/local/SSD_DEEPLEARNING_2' in pth:
                pth = pth.replace('/local/SSD_DEEPLEARNING_2/image_retrieval', os.path.expandvars('$SCRATCH'))

    if ('/local/SSD_DEEPLEARNING_' in pth) and (not os.path.exists('/local/SSD_DEEPLEARNING_1')):
        if os.path.exists('/local/DEEPLEARNING'):
            pth = pth.replace('/local/SSD_DEEPLEARNING_1', '/local/DEEPLEARNING')
            pth = pth.replace('/local/SSD_DEEPLEARNING_2', '/local/DEEPLEARNING')

        elif os.getenv('SCRATCH'):
            pth = pth.replace('/local/SSD_DEEPLEARNING_1', os.path.expandvars('$SCRATCH'))
            pth = pth.replace('/local/SSD_DEEPLEARNING_2', os.path.expandvars('$SCRATCH'))

    elif ('/share/DEEPLEARNING/datasets/image_retrieval' in pth) and (not os.path.exists('/share/DEEPLEARNING/datasets/image_retrieval')):
        if os.getenv('SCRATCH'):
            pth = pth.replace('/share/DEEPLEARNING/datasets/image_retrieval', os.path.expandvars('$SCRATCH'))

    elif ('gpfsscratch' in pth) and (not os.path.exists('/gpfsscratch/')):
        user = pth.split('/')[3]
        pattern = f"/gpfsscratch/rech/nfj/{user}"

        for replacement in [
            '/share/DEEPLEARNING/datasets/image_retrieval',
            '/local/DEEPLEARNING/image_retrieval',
            '/local/SSD_DEEPLEARNING_1/image_retrieval',
            '/local/SSD_DEEPLEARNING_2/image_retrieval',
        ]:
            tmp = pth.replace(pattern, replacement)
            if os.path.exists(tmp):
                pth = tmp
                break

    return pth
