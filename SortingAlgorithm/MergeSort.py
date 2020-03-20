def merge_sort(input_list):
    if len(input_list) < 2:
        return input_list
    mid_index = len(input_list)//2
    left = merge_sort(input_list[:mid_index])
    right = merge_sort(input_list[mid_index:])
    return merge(left, right)


def merge(left, right):
    i, j = 0, 0
    return_list = []
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            return_list.append(left[i])
            i += 1
        else:
            return_list.append(right[j])
            j += 1
    return_list = return_list + left[i:]
    return_list = return_list + right[j:]
    return return_list


if __name__ == "__main__":
    test_list = [1, 2, 4, 3, 9, 6, 7, 8]
    print(test_list)
    print(merge_sort(test_list))