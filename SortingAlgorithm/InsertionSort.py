def insertion_sort(input_list):
    for i in range(1, len(input_list)-1):
        j = i-1
        key = input_list[i]
        while j >=0 and key < input_list[j]:
            input_list[j+1] = input_list[j]
            j -= 1      
        input_list[j+1] = key

if __name__ == "__main__":
    test_list = [1, 2, 4, 3, 9, 6, 7, 8]
    print(test_list)
    insertion_sort(test_list)
    print(test_list)