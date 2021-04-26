names = input("get and process input for a list of names ").split(",")
assignments =  input("get and process input for a list of the number of assignments ").split(",")
grades =  input("get and process input for a list of grades ").split(",")


# message string to be used for each student
# HINT: use .format() with this string in your for loop

message = "Hi {},\n\nThis is a reminder that you have {} assignments left to submit before you can \
    graduate. You're current grade is {} and can increase to {} if you submit all assignments before \
    the due date.\n\n"

#for i in range(len(names)):
#    print(message.format(names[i], assignments[i], grades[i], int(grades[i]) + 2 * int(assignments[i])))

for name, assignment, grade in zip(names, assignments, grades):
    print(message.format(name, assignment, grade, int(grade) + int(assignment)*2))
