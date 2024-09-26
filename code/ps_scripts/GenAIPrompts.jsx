#target photoshop
//
// GenAIPrompts.jsx
//

//
// Generated Thu Feb 22 2024 13:52:25 GMT+0200
//

cTID = function(s) { return app.charIDToTypeID(s); };
sTID = function(s) { return app.stringIDToTypeID(s); };

//
// GenAIPrompts
//
//
//==================== GenerativeFillDog ==============
//

// default seed == -1 
function GenerativeFill(prompt_caption) {
  // Generative Fill
  function step1(enabled, withDialog) {
    if (enabled != undefined && !enabled)
      return;
    var dialogMode = (withDialog ? DialogModes.ALL : DialogModes.NO);
    var desc1 = new ActionDescriptor();
    var ref1 = new ActionReference();
    ref1.putEnumerated(cTID('Dcmn'), cTID('Ordn'), cTID('Trgt'));
    desc1.putReference(cTID('null'), ref1);
    desc1.putInteger(cTID('DocI'), 539);
    desc1.putInteger(cTID('LyrI'), 3);
    desc1.putString(sTID("prompt"), prompt_caption);
    desc1.putString(sTID("serviceID"), "clio");
    desc1.putString(sTID("workflow"), "in_painting");
    var desc2 = new ActionDescriptor();
    var desc3 = new ActionDescriptor();
    desc3.putString(sTID("gi_PROMPT"), prompt_caption);
    desc3.putString(sTID("gi_MODE"), "tinp");
    desc3.putInteger(sTID("gi_SEED"), -1);
    desc3.putInteger(sTID("gi_NUM_STEPS"), -1);
    desc3.putInteger(sTID("gi_GUIDANCE"), 6);
    desc3.putInteger(sTID("gi_SIMILARITY"), 0);
    desc3.putBoolean(sTID("gi_CROP"), false);
    desc3.putBoolean(sTID("gi_DILATE"), false);
    desc3.putInteger(sTID("gi_CONTENT_PRESERVE"), 0);
    desc3.putBoolean(sTID("gi_ENABLE_PROMPT_FILTER"), true);
    desc3.putBoolean(sTID("dualCrop"), true);
    desc3.putString(sTID("gi_ADVANCED"), "{\"enable_mts\":true}");
    desc2.putObject(cTID('clio'), cTID('clio'), desc3);
    desc1.putObject(sTID("serviceOptionsList"), cTID('null'), desc2);
    executeAction(sTID('syntheticFill'), desc1, dialogMode);
  };

  step1();      // Generative Fill
};

// EOF

"GenAIPrompts.jsx"
// EOF
