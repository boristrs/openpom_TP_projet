def choose_odor():
    print("Choose one or more of the odours you want to replace from the following options:")
    for i, value in enumerate(descriptors, start=1):
        print(f"{i}. {value}")

    while True :
      try:
        user_choice = input("Enter the numbers of the chosen scents, separated by commas (e.g. 1, 3): ")
        chosen_numbers = [int(number.strip()) for number in user_choice.split(',')]

        if all(1 <= number <= len(descriptors) for number in chosen_numbers):
          break
        else:
          print("Error: Please enter valid numbers within the range of available options.")
      except ValueError:
        print("Error: Please enter valid numbers within the range of available options.")

    chosen_values = [descriptors[number - 1] for number in chosen_numbers]

    print("You've chosen the following scents:")
    for value in chosen_values:
        print(value)

    return chosen_values

# Search for molecules containing exclusively these descriptors
def check_descriptor(row, target_list):
    descriptors = re.findall(r'([a-zA-Z]+)\s:', row)
    return set(descriptors) == set(target_list)
  
def only_odors(odors_description):
  # Create a Boolean series initially True for all lines
    all_present = pd.Series(True, index=df_descr.index)

  # Browse each odour in the odors_description list
    for o in odors_description:
      # Check the presence of the odour in the specific column
      presence_o = df_descr['Descriptors'].apply(lambda x: o in re.split(' : |, ', str(x)))
      # Update the Boolean series to indicate whether the smell is present in each line
      all_present = all_present & presence_o

  # Select the rows of the DataFrame where all odors in odors_description are present
    first_filter = df_descr[all_present]

  # Filter DataFrame rows according to condition
    first_filter = first_filter[first_filter['Descriptors'].apply(lambda row: check_descriptor(row, odors_description))]

    print('These molecules contain only the odours required: ')
    print(first_filter)
    return first_filter

#Search for molecules containing these descriptors and more
def at_least_odors(odors_description):
  # Create a Boolean series initially True for all lines
    all_present = pd.Series(True, index=df_descr.index)

  # Browse each odour in the odors_description list
    for o in odors_description:
      # Check the presence of the odour in the specific column
      presence_o = df_descr['Descriptors'].apply(lambda x: o in re.split(' : |, ', str(x)))
      # Update the Boolean series to indicate whether the smell is present in each line
      all_present = all_present & presence_o

  # Select the rows of the DataFrame where all odors in odors_description are present
    second_filter = df_descr[all_present]

    print('These molecules contain at least the n required odours:')
    print(second_filter)
    return second_filter

#Search for molecules containing at least one of the descriptors
def at_least_one_odor(odors_description):
  # Create a Boolean series initially True for all lines
    at_least_one = pd.Series(False, index=df_descr.index)

  # Browse each odour in the odors_description list
    for o in odors_description:
      # Check the presence of the odour in the specific column
      presence_o = df_descr['Descriptors'].apply(lambda x: o in re.split(' : |, ', str(x)))
      # Update the Boolean series to indicate whether at least one odour is present
      at_least_one = at_least_one | presence_o

  # Select the rows in the DataFrame where at least one of the odours is present
    third_filter = df_descr[at_least_one]

    print('These molecules contain at least one of the required odours:')
    print(third_filter)
    return third_filter
