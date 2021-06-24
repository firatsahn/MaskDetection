import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

def xml_to_csv(path):
    #global xml_list,xml_df,lenm
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            #lenm =[i for i in member]
            value = (root.find('filename').text,
                    int(root.find('size')[0].text),
                    int(root.find('size')[1].text),
                    root.find('object')[0].text,
                    int(member[5][0].text),
                    int(member[5][1].text),
                    int(member[5][2].text),
                    int(member[5][3].text))
 
            
            xml_list.append(value)
           
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    
    
    xml_df = pd.DataFrame(xml_list,columns=column_name)
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df
    #return xml_list

def main():
    for folder in ['train', 'test']:
        image_path = os.path.join(os.getcwd(), ('images/' + folder))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv(('images/'+folder+'_labels.csv'), index=None)
    print('Successfully converted xml to csv.')


main()
