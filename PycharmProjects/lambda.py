#A REGULAR FUNCTION
def guru( funct, *args ):
    funct( *args )

def printer_one( arg ):
    return print (arg)

def printer_two( arg ):
    print(arg)

#CALL A REGULAR FUNCTION
guru( printer_one, 'printer 1 REGULAR CALL' )
guru( printer_two, 'printer 2 REGULAR CALL \n' )

#CALL A REGULAR FUNCTION THRU A LAMBDA
guru(lambda: printer_one('printer 1 LAMBDA CALL'))
guru(lambda: printer_two('printer 2 LAMBDA CALL'))