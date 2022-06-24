from random import randint # (for sampling random integers)
from timeit import repeat #(for performing repeated timings)
import matplotlib.pyplot as plt #for plotting results
import numpy as np #for numerical tricks

def run_sorting_algorithm(sortlist,algorithm=sorted, batches=3,shots=10):

    # algorithm using the supplied list called list. Only import the
    # algorithm function if it's not the built-in `sorted()`.
    setup_code = f"from __main__ import {algorithm}" \
        if algorithm != "sorted" else ""

    stmt = f"{algorithm}({sortlist})"

    # Execute the code "number" different times in batchs of "repeat" and return the time
    # in seconds that each execution took where number and repeat are gotten from the variables passed in
    times = repeat(setup=setup_code, stmt=stmt, repeat=batches, number=shots)

    # minimum time it took to run

    print(f"Algorithm: {algorithm}. Minimum execution time: {min(times)} seconds")

size = 10

size_list = []
for x in range(size):
                
    size_list.append(randint(0, 1000))
#call run_sorting_algorithm
run_sorting_algorithm(size_list, 'sorted', 3, 10)

size_list_1 = []
size_list_2 = []
size_list_3 = []
for x in range(size):
    size_list_1.append(randint(0,1000))
    size_list_2.append(randint(0,1000))
    size_list_3.append(randint(0,1000))
run_sorting_algorithm(size_list_1, 'sorted', 3, 10)
run_sorting_algorithm(size_list_2, 'sorted', 3, 10)
run_sorting_algorithm(size_list_3, 'sorted', 3, 10)

def bubble_sort(list):
    n = len(list)
    for i in range(1, n):
        already_sorted = True
        for j in range(0,n-i):
            if list[j] > list[j+1]:
                temp = list[j+1]
                list[j+1] = list[j]
                list[j] = temp
                already_sorted = False
        if already_sorted == True:
            break
    return list

list = [8,2,6,4,5]
print (bubble_sort(list))
run_sorting_algorithm(list,"bubble_sort",1,1)

for i in range(1, 4):
    # set a length of list variable dependent on i
    
    #create your list of length 10^i random integers
    
    #Call run_sorting_algorithm
    
    bubble_results=[]

list_sizes=[10,100,1000,10000]

plt.rcParams["figure.figsize"]=(10,8)
plt.loglog(list_sizes, bubble_results,linestyle="None", marker='o',label="Data")
plt.loglog(list_sizes, np.power(np.list(list_sizes),0)*bubble_results[0]/10**0,label="$O(1)$")
plt.loglog(list_sizes, np.power(np.list(list_sizes),1)*bubble_results[0]/10**1,label="$O(n)$")
plt.loglog(list_sizes, np.power(np.list(list_sizes),2)*bubble_results[0]/10**2,label="$O(n^2)$")
plt.loglog(list_sizes, np.power(np.list(list_sizes),3)*bubble_results[0]/10**3,label="$O(n^3)$")
plt.legend()

def insertion_sort(sort_list):
 
    return sort_list

list = [8,2,6,4,5]
run_sorting_algorithm("insertion_sort",list,3,5)

for i in range(1, 4):
    # length of list
len(list) = 10^i     
   
insertion_results=[]

list_sizes=[10,100,1000,10000]

plt.rcParams["figure.figsize"]=(10,8)
plt.loglog(list_sizes, bubble_results,linestyle="None", marker='o',label="Bubble Data")
plt.loglog(list_sizes, insertion_results,linestyle="None", marker='o',label="Insertion Data")
plt.loglog(list_sizes, np.power(np.list(list_sizes),0)*insertion_results[0]/10**0,label="$O(1)$")
plt.loglog(list_sizes, np.power(np.list(list_sizes),1)*insertion_results[0]/10**1,label="$O(n)$")
plt.loglog(list_sizes, np.power(np.list(list_sizes),2)*insertion_results[0]/10**2,label="$O(n^2)$")
plt.loglog(list_sizes, np.power(np.list(list_sizes),3)*insertion_results[0]/10**3,label="$O(n^3)$")
plt.legend()

def merge(left, right):
    # If the first list is empty, then nothing needs
    # to be merged, and you can return the second list as the result
    if len(left) == 0:
        return right

    # If the second list is empty, then nothing needs
    # to be merged, and you can return the first list as the result
    if len(right) == 0:
        return left

    result = []
    index_left = index_right = 0

    # Now go through both lists until all the elements
    # make it into the resultant list
    while len(result) < len(left) + len(right):
        # The elements need to be sorted to add them to the
        # resultant list, so you need to decide whether to get
        # the next element from the first or the second list
        if left[index_left] <= right[index_right]:
            result.append(left[index_left])
            index_left += 1
        else:
            result.append(right[index_right])
            index_right += 1

        if index_right == len(right):
            result += left[index_left:]
            break

        if index_left == len(left):
            result += right[index_right:]
            break
return result

def merge_sort(list):
    if len(list) < 2:
        return list

    midpoint = len(list) // 2
    return merge(
        left=merge_sort(list[:midpoint]),
        right=merge_sort(list[midpoint:]))

for i in range(1,5):
    list_LENGTH=10**i

    list = [randint(0, 1000) for i in range(list_LENGTH)]

    run_sorting_algorithm("merge_sort",list,3,5)
    
    merge_results=[]
list_sizes=[10,100,1000,10000]

plt.rcParams["figure.figsize"]=(10,8)
plt.loglog(list_sizes, bubble_results,linestyle="None", marker='o',label="Bubble Data")
plt.loglog(list_sizes, insertion_results,linestyle="None", marker='o',label="Insertion Data")
plt.loglog(list_sizes, merge_results,linestyle="None", marker='o',label="Merge Data")
plt.loglog(list_sizes, np.power(np.list(list_sizes),0)*merge_results[0]/10**0,label="$O(1)$")
plt.loglog(list_sizes, np.power(np.list(list_sizes),1)*merge_results[0]/10**1,label="$O(n)$")
plt.loglog(list_sizes, np.power(np.list(list_sizes),1)*np.log2(np.list(list_sizes))*merge_results[0]/(10**1*np.log2(10)),label="$O(n\log_2(n))$")
plt.loglog(list_sizes, np.power(np.list(list_sizes),2)*merge_results[0]/10**2,label="$O(n^2)$")
plt.loglog(list_sizes, np.power(np.list(list_sizes),3)*merge_results[0]/10**3,label="$O(n^3)$")
plt.legend()
