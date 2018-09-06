with open('annotations/trainval.txt', 'w') as the_file:
    for x in range(1, 702):
        the_file.write('yellow_buoy_'+str(x)+' 1\n')
