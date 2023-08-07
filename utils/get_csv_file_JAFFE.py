def edit_JAFFE_csv_file_from_config(config, avatar_name):
    # modify csv according to avatar name
    if avatar_name in ['KA', 'KL', 'KM', 'KR', 'MK', 'NA', 'NM', 'TM', 'UY', 'YM']:
        config['train_csv'] = config['directory'] + f'/JAFFE_{avatar_name}_id.csv'
        config['test_csv'] = config['directory'] + f'/JAFFE_{avatar_name}_id.csv'
    else:
        raise ValueError("please select a valid avatar name", avatar_name)
