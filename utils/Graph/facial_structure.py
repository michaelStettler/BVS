import numpy as np


def control_face_struct(activities_dict, ref_struct):
    correct_face_struct = False
    correct_types = [ref_struct[i]["type"] for i in ref_struct]

    counter = np.zeros(np.max(correct_types) + 1)
    for i in activities_dict:
        idx = activities_dict[i]["type"]
        if idx in correct_types:
            counter[idx] += 1
        else:
            break

    # control that the dictionary has exactly one entry per node
    if np.sum(counter) == len(correct_types) and np.max(counter) == 1:
        correct_face_struct = True

    return correct_face_struct


def find_higest_actitivity_node(activities_dict):
    highest_activity = 0
    highest_idx = 0

    for i in activities_dict:
        activity = activities_dict[i]["max"]

        if activity > highest_activity:
            highest_idx = i
            highest_activity = activity

    return activities_dict[highest_idx]


def remove_activity_from_type(activities_dict, ref_type_dict):
    filtered_dict = {}

    for i in activities_dict:
        if activities_dict[i]["type"] != ref_type_dict["type"]:
            filtered_dict[i] = activities_dict[i]

    return filtered_dict


def infer_left_eye(face_pos1, face_pos2, ref_pos1, ref_pos2, prior_pos):
    # compute diff between nose of ref and face (need only vertical axis)
    diff_pos2X = ref_pos2[0] - face_pos2[0]

    # compute diff from left eye to ref according to diff of pos2
    diff_pos1X = ref_pos1[0] - face_pos1[0]

    # infer third positions
    return [prior_pos["pos"][0] + diff_pos1X - diff_pos2X, face_pos1[1]]


def infer_right_eye(face_pos1, face_pos2, ref_pos1, ref_pos2, prior_pos):
    # compute diff between nose of ref and face (need only vertical axis)
    diff_pos2X = ref_pos2[0] - face_pos2[0]

    # compute diff from left eye to ref according to diff of pos2
    diff_pos1X = ref_pos1[0] - face_pos1[0]

    # infer third positions
    return [prior_pos["pos"][0] + diff_pos1X - diff_pos2X, face_pos1[1]]


def infer_nose(face_pos1, face_pos2, ref_pos1, ref_pos2, prior_pos):
    print("[INFER_NOSE] TODO infer nose better!, so far simply return the prior since it is a the center of both eyes "
          "as the dataset is super simple...")
    return prior_pos["pos"]


def get_closest_lmk(activities_dict, ref_dict):
    closest_lmk = {"type": 0, "pos": [0, 0], "max": 0.0}
    min_dist = 1e10

    for i in activities_dict:
        dist = np.linalg.norm(np.array(activities_dict[i]["pos"]) - np.array(ref_dict["pos"]))
        if dist < min_dist:
            closest_lmk = activities_dict[i]
            min_dist = dist

    return closest_lmk


def fit_lmk_to_ref(face_struct, ref_struct, verbose=False):
    """
    Simply try to infer the third landmarks from the two first. The "graph" is a simple triangle for now

    :param face_struct:
    :param ref_struct:
    :return:
    """
    # retrieve positions of current face
    face_pos = [face_struct[0]["pos"], face_struct[1]["pos"]]
    # retrieve positions of reference face for same landmarks
    ref_pos = []
    lmk_found = np.zeros(len(ref_struct))
    for i in face_struct:
        lmk_type = face_struct[i]["type"]
        for j in ref_struct:
            if ref_struct[j]["type"] == lmk_type:
               ref_pos.append(ref_struct[j]["pos"])
               lmk_found[j] = 1

    # get prior (= missing lmk)
    lmk_to_infer = np.arange(len(ref_struct))
    lmk_to_infer = lmk_to_infer[lmk_found == 0]  # get all the non-found (=0) idx
    if len(lmk_to_infer) > 1:
        print("more than one landmark to infer!", lmk_to_infer)
    idx_to_infer = lmk_to_infer[0]  # simply get the first
    prior = ref_struct[idx_to_infer]

    # infer positions from the two known landmarks
    if face_struct[0]["type"] in [0, 3, 6] and face_struct[1]["type"] in [2, 5, 8]:
        # infer left eye
        lmk = infer_left_eye(face_pos[0], face_pos[1], ref_pos[0], ref_pos[1], prior)
    elif face_struct[1]["type"] in [0, 3, 6] and face_struct[0]["type"] in [2, 5, 8]:
        # infer left eye inverse
        lmk = infer_left_eye(face_pos[1], face_pos[0], ref_pos[1], ref_pos[0], prior)
    elif face_struct[0]["type"] in [0, 3, 6] and face_struct[1]["type"] in [1, 4, 7]:
        # infer nose
        lmk = infer_nose(face_pos[0], face_pos[1], ref_pos[0], ref_pos[1], prior)
    elif face_struct[1]["type"] in [0, 3, 6] and face_struct[0]["type"] in [1, 4, 7]:
        # infer nose inverse
        lmk = infer_nose(face_pos[1], face_pos[0], ref_pos[1], ref_pos[0], prior)
    elif face_struct[0]["type"] in [1, 4, 7] and face_struct[1]["type"] in [2, 5, 8]:
        # infer right eye
        lmk = infer_right_eye(face_pos[0], face_pos[1], ref_pos[0], ref_pos[1], prior)
    elif face_struct[1]["type"] in [1, 4, 7] and face_struct[0]["type"] in [2, 5, 8]:
        # infer right eye inverse
        lmk = infer_right_eye(face_pos[1], face_pos[0], ref_pos[1], ref_pos[0], prior)
    else:
        raise NotImplementedError("inferring landmarks from {} and {} is not yet implemented!".format(face_struct[0]["type"], face_struct[1]["type"]))

    if verbose:
        print("idx_to_infer", lmk_to_infer)
        print("prior:", prior)
        print("lmk_pos", lmk)

    return {"type": prior["type"], "pos": lmk, "max": 1.0}


def get_face_structure(activities_dict, ref_struct, verbose=False):
    """
    infer or clean landmarks to return a perfect face structure for one faace
    :return:
    """

    face_struc_dict = {}
    # remove entry which are not from this face type
    # that is because we already know the identity of the face here
    _activities_dict = {}
    lmk_types_oI = [ref_struct[i]["type"] for i in range(len(ref_struct))]
    for a in activities_dict:
        if activities_dict[a]["type"] in lmk_types_oI:
            _activities_dict[a] = activities_dict[a]

    # control if the activity dict has one node of each type
    is_correct_struct = control_face_struct(_activities_dict, ref_struct)

    if verbose:
        face_struct = []
        for i in _activities_dict:
            face_struct.append(_activities_dict[i]["type"])

    # if missing or having too many nodes
    if not is_correct_struct:
        # start with the highest activity and discard others form this type
        highest_activity_node = find_higest_actitivity_node(_activities_dict)
        filtered_activity = remove_activity_from_type(_activities_dict, highest_activity_node)
        face_struc_dict[0] = highest_activity_node

        # get the position of the highest activity from the second type
        highest_activity_2_node = find_higest_actitivity_node(filtered_activity)
        filtered_activity = remove_activity_from_type(filtered_activity, highest_activity_2_node)
        face_struc_dict[1] = highest_activity_2_node

        # infer third landmarks
        infer_struct_dict = fit_lmk_to_ref(face_struc_dict, ref_struct, verbose=verbose)

        # check if we need to infer or retrieve the closest activity
        if len(filtered_activity) == 0:
            # simply use the inferred third positions from first two points
            face_struc_dict[2] = infer_struct_dict
        else:
            # retrieve the closest landmark from the inference
            closest_lmk = get_closest_lmk(filtered_activity, infer_struct_dict)
            face_struc_dict[2] = closest_lmk

        if verbose:
            print("_activities_dict")
            print(_activities_dict)
            print("ref_struct")
            print(ref_struct)
            print("highest_activity_node", highest_activity_node)
            print("filtered_activity", filtered_activity)
            print("highest_activity_2_node", highest_activity_2_node)
            print("filtered_activity", filtered_activity)
            print("face_struc_dict", face_struc_dict)
            print("infer_struct_dict", infer_struct_dict)
    else:
        face_struc_dict = _activities_dict

    return face_struc_dict