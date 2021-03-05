"""
File: image_output_processing.py
Author: Shaurya Chandhoke
Description: Helper file which contains functions used for image outputting, image saving, as well as logging.
"""


def logger(step, complete, message, substep=False, *argv):
    """
    Global logging function that prints the status of the program to stdout.

    :param step: The step of the program it is currently in
    :param complete: Whether the step has started or completed
    :param message: A more verbose message indicating what's happening
    :param substep: Whether this step is a child step of a larger part of the program
    :param argv: Any extra information about the step
    """
    status = "Start" if complete == 0 else "Complete"
    finalMsg = "(Step {}) {}: {} ".format(step, status, message)

    if len(argv) != 0:
        finalMsg += "[ "
        for arg in argv:
            finalMsg += "{} ".format(arg)
        finalMsg += "]"

    if not substep:
        print(finalMsg)
    else:
        print("\t" + finalMsg)
