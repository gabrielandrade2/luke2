import random

import numpy as np

from scipy.stats import truncnorm


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def decision(probability):
    return random.random() < probability

def add_noise_char_simple(start, end, max_noise, stddev=1.5):
    normal = get_truncated_normal(mean=0, sd=stddev, low=0, upp=max_noise + 1)
    backward_noise = -int(normal.rvs())
    forward_noise = int(normal.rvs())
    if backward_noise:
        start += backward_noise
    if forward_noise:
        end += forward_noise
    return start, end


def add_noise_char(sentence, taglist, max_noise, stddev=1.5):
    out_tags = []
    for tag in taglist:
        start = tag[0]
        end = tag[1]
        if max_noise:
            if True:
                start, end = __add_noise_larger_char(start, end, max_noise, stddev)
            else:
                start, end = __add_noise_smaller_char(start, end, max_noise, stddev)
            out_tags.append((start, end, tag[2], sentence[start:end]))
    return out_tags


def __add_noise_larger_char(start, end, max_noise, stddev):
    normal = get_truncated_normal(mean=0, sd=stddev, low=0, upp=max_noise + 1)
    backward_noise = -int(normal.rvs())
    forward_noise = int(normal.rvs())
    if backward_noise:
        start += backward_noise
    if forward_noise:
        end += forward_noise
    return start, end


def __add_noise_smaller_char(start, end, max_noise, stddev):
    lenght = end - start
    if lenght <= 1:
        return start, end
    normal = get_truncated_normal(mean=0, sd=stddev, low=0, upp=min(max_noise, lenght) + 1)
    backward_noise = -int(normal.rvs())
    forward_noise = int(normal.rvs())
    if backward_noise:
        end += backward_noise
    if forward_noise:
        start += forward_noise
    return start, end


def add_noise_word(sentence, taglist, max_noise, stddev=1.2):
    out_tags = []
    for tag in taglist:
        start = tag[0]
        end = tag[1]
        if max_noise:
            if len(sentence[start:end].split(' ')) <= 1:
                choice = True
            else:
                choice = np.random.choice([True, False])
            if choice:
                temp = add_noise_larger_word(sentence, [tag], max_noise, stddev)[0]
            else:
                temp = __add_noise_smaller_word(sentence, [tag], max_noise, stddev)[0]
            start = temp[0]
            end = temp[1]
        out_tags.append((start, end, tag[2], sentence[start:end]))
    return out_tags


def add_noise_larger_word(sentence, taglist, max_noise, stddev=1.2):
    # normal = get_truncated_normal(mean=0, sd=1 + (max_noise/(1 + max_noise)), low=0, upp=max_noise + 1)
    normal = get_truncated_normal(mean=0, sd=stddev, low=0, upp=max_noise + 1)
    out_tags = []
    for tag in taglist:
        start = tag[0]
        end = tag[1]
        if max_noise:
            backward_noise = -int(normal.rvs())
            forward_noise = int(normal.rvs())
            if backward_noise:
                try:
                    start = get_word(sentence[:start], backward_noise)[0]
                except IndexError:
                    pass
            if forward_noise:
                try:
                    end = get_word(sentence[end:], forward_noise - 1)[1] + end
                except IndexError:
                    pass
        out_tags.append((start, end, tag[2], sentence[start:end]))
    return out_tags


def add_noise_larger_word_simple(sentence, s, e, max_noise, stddev=1.2):
    # normal = get_truncated_normal(mean=0, sd=1 + (max_noise/(1 + max_noise)), low=0, upp=max_noise + 1)
    normal = get_truncated_normal(mean=0, sd=stddev, low=0, upp=max_noise + 1)
    new_start = []
    new_end = []
    for start, end in zip(s, e):
        if max_noise:
            backward_noise = -int(normal.rvs())
            forward_noise = int(normal.rvs())
            if backward_noise:
                try:
                    start = get_word(sentence[:start], backward_noise)[0]
                except IndexError:
                    pass
            if forward_noise:
                try:
                    end += get_word(sentence[end:], forward_noise - 1)[1]
                except IndexError:
                    pass
            new_start.append(start)
            new_end.append(end)
    return new_start, new_end


def __add_noise_smaller_word(sentence, taglist, max_noise, stddev=1.2):
    # normal = get_truncated_normal(mean=0, sd=1 + (max_noise/(1 + max_noise)), low=0, upp=max_noise + 1)
    out_tags = []
    for tag in taglist:
        start = tag[0]
        end = tag[1]
        entity_length = len(sentence[start:end].split(' '))
        if entity_length <= 1:
            continue
        normal = get_truncated_normal(mean=0, sd=stddev, low=0, upp=min(max_noise, entity_length) + 1)
        if max_noise:
            backward_noise = -int(normal.rvs())
            forward_noise = int(normal.rvs())
            if (backward_noise + forward_noise) >= entity_length:
                if np.random.choice([True, False]):
                    forward_noise = 0
                else:
                    backward_noise = 0
            if backward_noise:
                try:
                    end = start + get_word(sentence[start:end], backward_noise)[0] - 1  # To account for spaces
                except IndexError:
                    pass
            if forward_noise:
                try:
                    start = start + get_word(sentence[start:end], forward_noise)[0]
                except IndexError:
                    pass
        out_tags.append((start, end, tag[2], sentence[start:end]))
    return out_tags


def get_word(str, target_index):
    start = len(" ".join(list(filter(bool, str.split(" ")))[:target_index]))
    # Compensate the initial space
    if str.startswith(' '):
        start += 1
    # Compensate the space between the last word and before the target one
    while str[start] == ' ':
        start += 1

    end = start + len(list(filter(bool, str.split(' ')))[target_index])
    end += str[start:end].count(' ')
    return start, end


def add_noise_larger_single_word_probability(sentence, s, e, noise_ratio):
    new_start = []
    new_end = []
    for start, end in zip(s, e):
        if decision(noise_ratio):
            # 1 - backward_noise, 2 - forward_noise, 3 - both
            noise_type = np.random.choice([1, 2, 3])
            if noise_type & 1:
                try:
                    start = get_word(sentence[:start], -1)[0]
                except IndexError:
                    pass
            if noise_type & 2:
                try:
                    end += get_word(sentence[end:], 1 - 1)[1]
                except IndexError:
                    pass
        new_start.append(start)
        new_end.append(end)
    return new_start, new_end
