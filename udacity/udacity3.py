x = 1
while True:
   try:
      x = int(input("Enter a number: "))
      # print(x)
      break
   except ValueError:
      print("that\'s not a walid number\n")
   except KeyboardInterrupt:
      print("No onput tekken\n")
      break
   finally:
      print(x)
      print("zawsze\n")
