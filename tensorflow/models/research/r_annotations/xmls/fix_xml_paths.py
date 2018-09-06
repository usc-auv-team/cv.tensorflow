import xml.etree.ElementTree as ET
from PIL import Image

# for x in range (1, 702):
#     try:
#         with open('yellow_buoy_'+str(x)+'.xml') as f:
#           tree = ET.parse(f)
#           root = tree.getroot()
#
#           for path in root.iter('path'):
#             try:
#             #PUT YOUR OWN PATH HERE
#               path.text = '/home/pedro/usc-computer-vision-2018/tensorflow/models/research/images/yellow_buoy_' + str(x) + '.jpg'
#             except AttributeError:
#               pass
#
#         tree.write('yellow_buoy_'+str(x)+'.xml')
#     except IOError:
#         pass
#
#
#filename fixe
# for x in range (1, 702):
#     try:
#         with open('yellow_buoy_'+str(x)+'.xml') as f:
#           tree = ET.parse(f)
#           root = tree.getroot()
#
#           for path in root.iter('folder'):
#             try:
#               path.text = 'images'
#             except AttributeError:
#               pass
#
#         tree.write('yellow_buoy_'+str(x)+'.xml')
#     except IOError:
#         pass
#
# for x in range (1, 702):
#     try:
#         with open('yellow_buoy_'+str(x)+'.xml') as f:
#           tree = ET.parse(f)
#           root = tree.getroot()
#
#           for path in root.iter('filename'):
#             try:
#               path.text = 'yellow_buoy_'+ str(x) + '.jpg'
#             except AttributeError:
#               pass
#
#         tree.write('yellow_buoy_'+str(x)+'.xml')
#     except IOError:
#         pass

# for x in range (1, 702):
#     try:
#         with open('yellow_buoy_'+str(x)+'.xml') as f:
#           tree = ET.parse(f)
#           root = tree.getroot()
#
#           for name in root.iter('name'):
#             try:
#                 if name.text == 'yellow_buoy':
#                     name.text = 'buoy_yellow'
#             except AttributeError:
#               pass
#
#         tree.write('yellow_buoy_'+str(x)+'.xml')
#     except IOError:
#         pass

# for x in range (1, 1714):
#     try:
#         with open(str(x)+'.xml') as f:
#           tree = ET.parse(f)
#           root = tree.getroot()
#           child = ET.Element('filename')
#           root.append(child)
#           #
#           # for name in root.iter('name'):
#           #   try:
#           #       if name.text == 'yellow_buoy':
#           #           name.text = 'buoy_yellow'
#           #   except AttributeError:
#           #     pass
#
#         tree.write(str(x)+'.xml')
#     except IOError:
#         pass

# for x in range (1, 1714):
#     try:
#         with open(str(x)+'.xml') as f:
#           tree = ET.parse(f)
#           root = tree.getroot()
#
#           for filename in root.iter('filename'):
#             try:
#                 filename.text = str(x) + '.jpg'
#             except AttributeError:
#               pass
#
#         tree.write(str(x)+'.xml')
#     except IOError:
#         pass

# DELETES SIZE TAGS
for x in range (1, 1714):
    try:
        img = Image.open('../../images/'+str(x)+'.jpg')
        img_width, img_height, = img.size

        with open(str(x)+'.xml') as f:
            tree = ET.parse(f)
            root = tree.getroot()

            for size in root.findall('size'):
                root.remove(size)
                # if child.name != "size":
                #     continue
                # else:
                #     root.remove(child)

            tree.write(str(x)+'.xml')
    except IOError:
        pass

# ADDS HEIGHT AND WIDTH TO XML FILES
for x in range (1, 1714):
    try:
        img = Image.open('../../images/'+str(x)+'.jpg')
        img_width, img_height, = img.size

        with open(str(x)+'.xml') as f:
            tree = ET.parse(f)
            root = tree.getroot()
            child = ET.Element('size')
            root.append(child)
            child.append(ET.Element('height'))
            child.append(ET.Element('width'))

            for height in root.iter('height'):
                try:
                    height.text = str(img_height)
                except AttributeError:
                  pass

            for width in root.iter('width'):
                try:
                    width.text = str(img_width)
                except AttributeError:
                  pass


            tree.write(str(x)+'.xml')
    except IOError:
        pass

# ADDS DIFFICULT 0 TO XMLS
# for x in range (1, 1714):
#     try:
#         with open(str(x)+'.xml') as f:
#             tree = ET.parse(f)
#             root = tree.getroot()
#             child = ET.Element('difficult')
#
#             for object in root.iter('object'):
#                 try:
#                     object.append(child)
#
#                 except AttributeError:
#                     pass
#
#             for diff in root.iter('difficult'):
#                 try:
#                     diff.text = '0'
#
#                 except AttributeError:
#                     pass
#
#             tree.write(str(x)+'.xml')
#     except IOError:
#         pass

# ADDS TRUNCATED TO XMLS
# for x in range (1, 1714):
#     try:
#         with open(str(x)+'.xml') as f:
#             tree = ET.parse(f)
#             root = tree.getroot()
#             child = ET.Element('truncated')
#
#             for object in root.iter('object'):
#                 try:
#                     object.append(child)
#
#                 except AttributeError:
#                     pass
#ls
#             for truncated in root.iter('truncated'):
#                 try:
#                     truncated.text = '0'
#
#                 except AttributeError:
#                     pass
#
#             tree.write(str(x)+'.xml')
#     except IOError:
#         pass


# ADDS pose
# for x in range (1, 1714):
#     try:
#         with open(str(x)+'.xml') as f:
#             tree = ET.parse(f)
#             root = tree.getroot()
#             child = ET.Element('pose')
#
#             for object in root.iter('object'):
#                 try:ls
#                     object.append(child)
#
#                 except AttributeError:
#                     pass
#
#             for pose in root.iter('pose'):
#                 try:
#                     pose.text = 'Unspecified'
#
#                 except AttributeError:
#                     pass
#
#             tree.write(str(x)+'.xml')
#     except IOError:
#         pass
