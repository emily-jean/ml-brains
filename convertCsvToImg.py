#!/usr/bin/env python3
import fileinput

c = 0
for line in fileinput.input():
    pixels = line.rstrip().split(',')
    if pixels[3087] == '1':
        c += 1
    if pixels[3087] != '1' or c < 3:
        continue
    with open('slices3.pgm', 'w') as out:
        out.write('P2\n')
        out.write('21 147\n')
        out.write('255\n')
        for k in range(7):
            for i in range(21):
                for j in range(21):
                    out.write(pixels[21 * i + j + k * 21 * 21])
                    out.write(' ')
                out.write('\n')
    break
