import xml.etree.ElementTree as ET

for x in range (1, 702):
    try:
        with open(str(x)+'.xml') as f:
          tree = ET.parse(f)
          root = tree.getroot()

          for name in root.iter('name'):
            try:
                if name.text == 'path_marker':
                    print(str(x))
            except AttributeError:
              pass

        tree.write('yellow_buoy_'+str(x)+'.xml')
    except IOError:
        pass
