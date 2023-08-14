// Test try catch



try {
    console.log("Start of try runs");

    unicycle;
    console.log("End of try runs -- never reached")
}

catch(error){
    console.log("Error has occured: " + error.stack);
}

finally{
    console.log("This is always run")
}

console.log("... Then the execution continues")
