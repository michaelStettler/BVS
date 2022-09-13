def edit_FERG_csv_file_from_config(config, avatar_name):
    # modify csv according to avatar name
    if avatar_name == 'jules':
        config['train_csv'] = config['train_csv'] + '_jules.csv'
        config['test_csv'] = config['test_csv'] + '_jules.csv'
    elif avatar_name == 'malcolm':
        config['train_csv'] = config['train_csv'] + '_malcolm.csv'
        config['test_csv'] = config['test_csv'] + '_malcolm.csv'
    elif avatar_name == 'ray':
        config['train_csv'] = config['train_csv'] + '_ray.csv'
        config['test_csv'] = config['test_csv'] + '_ray.csv'
    elif avatar_name == 'aia':
        config['train_csv'] = config['train_csv'] + '_aia.csv'
        config['test_csv'] = config['test_csv'] + '_aia.csv'
    elif avatar_name == 'bonnie':
        config['train_csv'] = config['train_csv'] + '_bonnie.csv'
        config['test_csv'] = config['test_csv'] + '_bonnie.csv'
    elif avatar_name == 'mery':
        config['train_csv'] = config['train_csv'] + '_mery.csv'
        config['test_csv'] = config['test_csv'] + '_mery.csv'
    else:
        raise ValueError("please select a valid avatar name", avatar_name)
