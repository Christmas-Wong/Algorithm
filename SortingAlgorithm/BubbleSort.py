def bubble_sort(input_list):
    i = 0
    while i < len(input_list):
        j = 0
        while j < len(input_list) - i - 1:
            if input_list[j] > input_list[j+1]:
                mid_num = input_list[j]
                input_list[j] = input_list[j+1]
                input_list[j+1] = mid_num
            j += 1
        i += 1


if __name__ == '__main__':
    test_list = [1, 2, 4, 3, 9, 6, 7, 8]
    print(test_list)
    bubble_sort(test_list)
    print(test_list)