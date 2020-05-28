import argparse
import sys

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('columns', type=int)
  args = parser.parse_args()

  row = []
  data = []  
  while True:
    line = sys.stdin.readline().strip()
    if not line:
      break
    row.append(line)
    if len(row) >= args.columns:
      data.append(row)
      row = []
  if row:
    data.append(row)
  
  for row in data:
    print(','.join(row))
    
  
