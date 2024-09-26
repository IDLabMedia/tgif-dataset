#include "GenAIPrompts.jsx"
#include "ActionFileFromXMLWithoutPopup.jsx"

const DO_LOGGING = true;
//const DO_LOGGING = false;

const SKIP_IF_EXISTS = true;
//const SKIP_IF_EXISTS = false;

//const WAIT_BETWEEN_SAVES = 3000;
const WAIT_BETWEEN_SAVES = 1000;

//const WAIT_BETWEEN_IMAGES = 1000;
const WAIT_BETWEEN_IMAGES = 8000;


//main_dialogs();
main_window();

function main_window() {
    var process_function = process;
    createDialog(process);
}

function main_dialogs() {
    // Ask user to select the base folder
    var baseFolder = Folder.selectDialog("Select base folder that contain all orig and mask images");
    // Ask user to select the CSV file
    var csvFile = File.openDialog("Select the CSV file (origfile,maskfile,prompt)", "*.csv");
    // Ask user to select output folder
    var outputFolder = Folder.selectDialog("Select output folder");

    if (baseFolder != null && outputFolder != null && csvFile != null) {
        process(baseFolder, outputFolder, csvFile);
        alert("Script completed successfully!");
    } else {
        alert("No Base Folder, Output folder, or CSV file selected. Script aborted.");
    }
}

function process(baseFolder, outputFolder, csvFile, doLogging, skipIfExists) {
    // Read the CSV file
    csvFile.open("r");
    var csvContent = csvFile.read();
    csvFile.close();
    var separator = ",";
    var rows = csvContent.split("\n");

    if(doLogging) {
        // Generate a unique filename based on the current timestamp
        var logFileName = "log_" + getTimestampString() + ".log";
        var logFilePath = outputFolder + "/" + logFileName;
        var logFile = new File(logFilePath);
        var logMessages = [
            "Starting GenAIBatch using parameters:",
            "baseFolder: " + baseFolder,
            "csvFile: " + csvFile,
            "outputFolder: " + outputFolder
        ]
        writeLogs(logMessages, logFile);
    }

    if (rows.length > 1) { // Check if there is at least a header
        var rows_to_process = rows.length - 1;
        //rows_to_process = 1; // For debugging: only do with first
        for (var i = 1; i < 1 + rows_to_process; i++) { // Iterate over rows (start from 1 to skip the header)
            var row = rows[i].split(separator);

            if(row.length >= 3) { // ignore incomplete or empty lines
                // Extract orig, mask, and prompt from the csv-row
                var origPath = baseFolder + "/" + row[0];
                var maskPath = baseFolder + "/" + row[1];
                var prompt_caption = row[2];

                if(doLogging) { writeLog("Iteration #" + i + " (" + row[0] + ", "+  row[1] + ", " + row[2] + ")" + " at " + getTimestampString(), logFile); }

                if(skipIfExists) {
                    var maskFile = new File(maskPath);
                    var maskFileName = maskFile.name;
                    var resultFileNamePrefix = maskFileName + "_ps_";
                    var newMaskFilePromptFolder = new Folder(outputFolder + "/" + prompt_caption + "/");
                    if (!newMaskFilePromptFolder.exists) {
                        newMaskFilePromptFolder.create();
                    }
                    var newMaskFile = new File(newMaskFilePromptFolder + "/" + resultFileNamePrefix + "mask.png");
                    if(newMaskFile.exists) {
                        writeLog("Skipping", logFile);
                        continue;
                    }

                }
                if(1 < i && i < 1 + rows_to_process + 1) {
                    $.sleep(WAIT_BETWEEN_IMAGES); // Sleep to prevent Photoshop from banning me (in addition to sleep between saves)
                }
                try {
                    processTuple(origPath, maskPath, prompt_caption, outputFolder);   
                } catch(error) {
                    alert("Error " + error.message);
                    if (error.message === "InvalidMask") {
                        alert("Invalid Mask! Skipping this image."); // This custom error is never thrown anymore
                    } else {
                        alert("An error occurred: " + error.message);
                        throw error
                    }
                }
            }
        }
    }
}

function getTimestampString() {
    var now = new Date();
    var year = now.getFullYear();
    var month = padZero(now.getMonth() + 1);
    var day = padZero(now.getDate());
    var hours = padZero(now.getHours());
    var minutes = padZero(now.getMinutes());
    var seconds = padZero(now.getSeconds());
    var timestampString = year + "_" + month + "_" + day + "_" + hours + "_" + minutes + "_" + seconds;

    return timestampString;
}

// Function to pad single-digit numbers with leading zero
    function padZero(num) {
        return num < 10 ? "0" + num : num;
    }

function writeLog(message, logFile) {
    writeLogs([message], logFile);
}

// Function to write messages to a log file
function writeLogs(messages, logFile) {
    logFile.open("a"); // Open the file in append mode
    
    // Write each message to the log file
    for (var i = 0; i < messages.length; i++) {
        logFile.writeln(messages[i]);
    }
    
    logFile.close(); // Close the file
}

const seed = -1; // -1 is no seed
//const seed = 26091993;

function processTuple(origPath, maskPath, prompt_caption, outputFolder) {
    // Open the orig and mask images
    var origFile = new File(origPath);
    var maskFile = new File(maskPath);

    // Add the mask as a new layer onto copy of original
    var maskDoc = open(maskFile);
    maskDoc.selection.selectAll();
    maskDoc.selection.copy();
    var origDoc = open(origFile);
    origDoc.paste();
    maskDoc.close(SaveOptions.DONOTSAVECHANGES);

    var newDoc = origDoc;

    // Actions: "SelectMask", SwitchLayers
    app.doAction("SelectMask", "GenAI");
    app.doAction("SwitchLayers", "GenAI");

    // Generate Fill
    GenerativeFill(prompt_caption);

    // Save all variations
    // Save variation 1
    var maskFileName = maskFile.name;
    var resultFileNamePrefix = maskFileName + "_ps_";
    var savePath = new File(outputFolder + "/" + prompt_caption + "/" + resultFileNamePrefix + "0.png");
    newDoc.saveAs(savePath, new PNGSaveOptions(), true);

    // Select & Save Variation 2
    $.sleep(WAIT_BETWEEN_SAVES); // Sleep to prevent blocking of my laptop
    app.doAction("SelectVariation2", "GenAI");
    savePath = new File(outputFolder + "/" + prompt_caption + "/" + resultFileNamePrefix + "1.png");
    newDoc.saveAs(savePath, new PNGSaveOptions(), true);

    // Select & Save Variation 3
    $.sleep(WAIT_BETWEEN_SAVES); // Sleep to prevent blocking of my laptop
    app.doAction("SelectVariation3", "GenAI");
    savePath = new File(outputFolder + "/" + prompt_caption + "/" + resultFileNamePrefix + "2.png");
    newDoc.saveAs(savePath, new PNGSaveOptions(), true);

    // Save new mask
    app.doAction("MakeMask", "GenAI");
    $.sleep(WAIT_BETWEEN_SAVES); // Sleep to prevent blocking of my laptop
    savePath = new File(outputFolder + "/" + prompt_caption + "/" + resultFileNamePrefix + "mask.png");
    newDoc.saveAs(savePath, new PNGSaveOptions(), true);

    // Close the document
    newDoc.close(SaveOptions.DONOTSAVECHANGES);
}

// Function to create a row with a static text field and an edit text field
function createRow(parent, labelText, defaultPath, isFolder) {
    var row = parent.add("group");
    row.orientation = "row";

    var staticTextGroup = row.add("group");
    staticTextGroup.orientation = "column";
    staticTextGroup.alignment = "right";
    var staticText = staticTextGroup.add("statictext", undefined, labelText);
    staticText.preferredSize.width = 80;
    staticText.alignment = "right";

    var editTextGroup = row.add("group");
    editTextGroup.orientation = "column";
    editTextGroup.alignment = "left";
    var editText = editTextGroup.add("edittext", undefined, defaultPath, {multiline: false, scrolling: true});
    editText.characters = 50;

    var buttonGroup = row.add("group");
    buttonGroup.orientation = "column";
    buttonGroup.alignment = "left";
    var button = buttonGroup.add("button", undefined, "...");
    button.onClick = function() {
        if(isFolder) {
            var folderPath = editText.text;
            var parentFolder = Folder(folderPath).parent;
            var selectedFolder = parentFolder.selectDlg("Select folder");
        } else {
            var fsel = Stdlib.createFileSelect("Csv Files: *.csv,All Files:*");
            var selectedFolder = Stdlib.selectFileOpen(labelText, fsel, editText.text);
        }

        if (selectedFolder) {
            editText.text = selectedFolder.fsName;
        }
    };
    
    return editText;
}



function createDialog(process_function) {
    // Default paths
    var defaultBaseFolderPath = "C:\\w\\Experiments\\2024-02 tgiif\\experiments";
    var defaultCsvFilePath = "C:\\w\\Experiments\\2024-02 tgiif\\ps_experiments\\csv\\all_others_interleaved.csv";
    var defaultOutputFolderPath = "C:\\w\\Experiments\\2024-02 tgiif\\ps_experiments\\output\\all_others_interleaved";

    // Create a window
    var dialog = new Window("dialog", "Path Selection");
    dialog.orientation = "column";

    // Add rows for base folder, CSV file, and output folder
    var baseFolderText = createRow(dialog, "Base Folder:", defaultBaseFolderPath, true);
    var csvFileText = createRow(dialog, "CSV File:", defaultCsvFilePath, false);
    var outputFolderText = createRow(dialog, "Output Folder:", defaultOutputFolderPath, true);

    // OK and Cancel buttons
    var buttonGroup = dialog.add("group");
    var goButton = buttonGroup.add("button", undefined, "GO");
    var cancelButton = buttonGroup.add("button", undefined, "Cancel");

    // Button click event handlers
    goButton.onClick = function() {
        // Get the selected paths
        var baseFolderPath = baseFolderText.text;
        var baseFolder = new Folder(baseFolderPath);
        var csvFilePath = csvFileText.text;
        var csvFile = new File(csvFilePath);
        var outputFolderPath = outputFolderText.text;
        var outputFolder = new Folder(outputFolderPath);

        if (!outputFolder.exists) {
            outputFolder.create();
        }

        dialog.close();
        try {
            process(baseFolder, outputFolder, csvFile, DO_LOGGING, SKIP_IF_EXISTS);
        } catch (e) {
            alert("An error occurred: " + e);
        }
    };
    cancelButton.onClick = function() {
        // Close the dialog without doing anything
        dialog.close();
    };

    // Show the dialog
    dialog.show();
}