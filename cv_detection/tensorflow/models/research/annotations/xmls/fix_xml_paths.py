import xml.etree.ElementTree as ET

for x in range (1, 702):
    try:
        with open('yellow_buoy_'+str(x)+'.xml') as f:
          tree = ET.parse(f)
          root = tree.getroot()

          for path in root.iter('path'):
            try:
            #PUT YOUR OWN PATH HERE
              path.text = '/home/pedro/usc-computer-vision-2018/tensorflow/models/research/images/yellow_buoy_' + str(x) + '.jpg'
            except AttributeError:
              pass

        tree.write('yellow_buoy_'+str(x)+'.xml')
    except IOError:
        pass


#filename fixe
for x in range (1, 702):
    try:
        with open('yellow_buoy_'+str(x)+'.xml') as f:
          tree = ET.parse(f)
          root = tree.getroot()

          for path in root.iter('folder'):
            try:
              path.text = 'images'
            except AttributeError:
              pass

        tree.write('yellow_buoy_'+str(x)+'.xml')
    except IOError:
        pass

for x in range (1, 702):
    try:
        with open('yellow_buoy_'+str(x)+'.xml') as f:
          tree = ET.parse(f)
          root = tree.getroot()

          for path in root.iter('filename'):
            try:
              path.text = 'yellow_buoy_'+ str(x) + '.jpg'
            except AttributeError:
              pass

        tree.write('yellow_buoy_'+str(x)+'.xml')
    except IOError:
        pass
