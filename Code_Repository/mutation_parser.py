import pandas as pd
import xlsxwriter
from collections import defaultdict

#mut = pd.read_excel('../../Documents/mutations2.xlsx', header=0)
mut = pd.read_excel('../data/mutations.xlsx', 1)
#21002 transcripts
#1001 cosmic ids


transcripts = mut.iloc[:,4]
cosmics = mut.iloc[:,1]

mappings = defaultdict(list)

for i in range(len(transcripts)):
    mappings[transcripts[i]].append(cosmics[i])


t = len(set(transcripts))
c = len(set(cosmics))


trans_set = set(transcripts)
cosm_set = set(cosmics)

trans_set = sorted(list(trans_set))
cosm_set = sorted(list(cosm_set))


#print(trans_set)
#print(cosm_set)


#trans x cosmic
mat = [0] * c

for i in range(c):
    mat[i] = [0] * t

#print(mat)

#first line = cosmics
#first col = transcripts
#mat.iloc[0] = cosmics
#mat[:,0] = transcripts

for tr in mappings:
    for co in mappings[tr]:
        posx = trans_set.index(tr)
        posy = cosm_set.index(co)
        mat[posy][posx] = 1

#print(mat)


workbook = xlsxwriter.Workbook('newmut.xlsx')
worksheet = workbook.add_worksheet()


row = 1
for col, data in enumerate(mat):
    worksheet.write_column(row, col+1, data)

#write name list
worksheet.write_row(0, 1, cosm_set)
worksheet.write_column(1, 0, trans_set)

workbook.close()