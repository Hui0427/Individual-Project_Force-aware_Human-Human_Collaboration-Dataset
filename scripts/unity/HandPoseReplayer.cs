// HandPoseReplayer
//
// Setup: drop this script + a replayer_*.json onto the person's FBX instance root
// (the object that has the Animator), assign the json to Pose Json, press Play.
//
// The Animator keeps driving the body and root motion; this script only overrides the
// hand bones, so the character cannot drift or change position because of it.
//
// A/B compare: toggle "Play Refined" during playback. For a still comparison tick
// "Freeze" and scrub "Freeze Frame"; the whole character holds on that frame.
using System;
using System.Collections.Generic;
using UnityEngine;

public class HandPoseReplayer : MonoBehaviour
{
    [Header("Data")]
    [Tooltip("replayer_<person>.json produced by export_replayer_json.py")]
    public TextAsset poseJson;

    [Header("Compare")]
    [Tooltip("ON = optimised hands, OFF = original Captury hands. Toggle while playing.")]
    public bool playRefined = true;

    [Header("Freeze / scrub")]
    [Tooltip("Hold the whole character on Freeze Frame so poses can be compared as stills.")]
    public bool freeze = false;
    [Tooltip("Frame to hold. High-difference frames are listed in skeleton_meta.json.")]
    public int freezeFrame = 0;

    [Header("Display")]
    public bool showOverlay = true;
    [Tooltip("Frame range that was optimised; highlighted in the overlay.")]
    public int refinedStart = 0;
    public int refinedEnd = 99999;

    [Header("Advanced")]
    [Tooltip("Right-handed source to left-handed Unity. Turn off only if hands look mirrored.")]
    public bool mirrorRotations = true;
    [Tooltip("Diagnostic: wiggle every bound hand bone. Nothing moves = names did not match.")]
    public bool testWiggle = false;

    [Serializable]
    class PoseData
    {
        public float fps;
        public int frames;
        public string[] names;
        public string hipsName;
        public float[] qOrig;
        public float[] q;
        public float[] hipsT;
    }

    PoseData data;
    readonly Dictionary<string, Transform> bones = new Dictionary<string, Transform>();
    readonly List<int> handIdx = new List<int>();
    Animator anim;
    float fallbackT0;
    int curFrame;
    int matched;

    // Freeze bookkeeping: seek the Animator only when the target frame actually changes,
    // otherwise the re-seek every LateUpdate fights anim.speed = 0 and nothing holds.
    bool wasFrozen;
    int seekedFrame = -1;
    float savedSpeed = 1f;

    GUIStyle style;

    void Start()
    {
        if (poseJson == null)
        {
            Debug.LogError("HandPoseReplayer: no Pose Json assigned.");
            enabled = false;
            return;
        }
        data = JsonUtility.FromJson<PoseData>(poseJson.text);
        foreach (var tr in GetComponentsInChildren<Transform>(true))
            if (!bones.ContainsKey(tr.name)) bones[tr.name] = tr;

        for (int n = 0; n < data.names.Length; n++)
            if (data.names[n].Contains("Hand"))
                handIdx.Add(n);

        var missing = new List<string>();
        foreach (int n in handIdx)
            if (bones.ContainsKey(data.names[n])) matched++;
            else if (missing.Count < 5) missing.Add(data.names[n]);

        anim = GetComponent<Animator>();   // left enabled: it still drives body + root
        if (anim != null) savedSpeed = anim.speed;
        fallbackT0 = Time.time;

        Debug.Log($"HandPoseReplayer: {matched}/{handIdx.Count} hand bones bound, " +
                  $"{data.frames} frames @ {data.fps} fps");
        if (matched < handIdx.Count)
            Debug.LogWarning("HandPoseReplayer: unmatched bone names, e.g. [" +
                             string.Join(", ", missing) + "] - check the rig's naming.");
    }

    void OnDisable()
    {
        if (anim != null) anim.speed = savedSpeed;   // never leave the Animator stopped
    }

    void LateUpdate()
    {
        if (data == null) return;
        bool hasAnim = anim != null && anim.enabled && anim.runtimeAnimatorController != null;
        freezeFrame = Mathf.Clamp(freezeFrame, 0, data.frames - 1);

        if (freeze)
        {
            curFrame = freezeFrame;
            if (hasAnim)
            {
                if (!wasFrozen) { savedSpeed = anim.speed; anim.speed = 0f; seekedFrame = -1; }
                if (seekedFrame != curFrame)
                {
                    var st = anim.GetCurrentAnimatorStateInfo(0);
                    if (st.length > 0f)
                    {
                        anim.Play(st.fullPathHash, 0, (curFrame / data.fps) / st.length);
                        anim.Update(0f);            // apply the seek immediately
                    }
                    seekedFrame = curFrame;
                }
            }
        }
        else
        {
            if (hasAnim && wasFrozen) { anim.speed = savedSpeed <= 0f ? 1f : savedSpeed; }
            seekedFrame = -1;

            float t;
            if (hasAnim)
            {
                var st = anim.GetCurrentAnimatorStateInfo(0);
                t = Mathf.Repeat(st.normalizedTime, 1f) * st.length;
            }
            else
            {
                t = Mathf.Repeat(Time.time - fallbackT0, data.frames / data.fps);
            }
            curFrame = Mathf.Clamp((int)(t * data.fps), 0, data.frames - 1);
        }
        wasFrozen = freeze;

        var Q = playRefined ? data.q : data.qOrig;
        int nb = data.names.Length;
        foreach (int n in handIdx)
        {
            Transform tr;
            if (!bones.TryGetValue(data.names[n], out tr)) continue;
            int i = (curFrame * nb + n) * 4;
            var q = mirrorRotations
                ? new Quaternion(Q[i], -Q[i + 1], -Q[i + 2], Q[i + 3])
                : new Quaternion(Q[i], Q[i + 1], Q[i + 2], Q[i + 3]);
            if (testWiggle)
                q *= Quaternion.Euler(Mathf.Sin(Time.time * 6f) * 40f, 0, 0);
            tr.localRotation = q;
        }
    }

    string Status()
    {
        string mode = playRefined ? "REFINED" : "ORIGINAL";
        string seg = (curFrame >= refinedStart && curFrame < refinedEnd)
            ? "   << optimised range" : "";
        string frz = freeze ? "   [FROZEN]" : "";
        return $"t = {curFrame / data.fps:F2}s   frame {curFrame}   [{mode}]{frz}{seg}";
    }

    void OnGUI()
    {
        if (data == null || !showOverlay) return;
        if (style == null)
            style = new GUIStyle(GUI.skin.label) { fontSize = 20, fontStyle = FontStyle.Bold };
        style.normal.textColor = playRefined ? Color.cyan : Color.white;
        GUI.Label(new Rect(12, 8, 900, 30), Status(), style);
        GUI.Label(new Rect(12, 32, 900, 30),
                  $"{matched} hand bones bound   |   Play Refined = A/B compare", style);
    }

#if UNITY_EDITOR
    void OnDrawGizmos()   // same readout in the Scene view, above the character
    {
        if (!Application.isPlaying || data == null || !showOverlay) return;
        Transform hips;
        var anchor = bones.TryGetValue(data.hipsName, out hips)
            ? hips.position : transform.position;
        var s = new GUIStyle { fontSize = 15, fontStyle = FontStyle.Bold };
        s.normal.textColor = playRefined ? Color.cyan : Color.white;
        UnityEditor.Handles.Label(anchor + Vector3.up * 1.2f, Status(), s);
    }
#endif
}
