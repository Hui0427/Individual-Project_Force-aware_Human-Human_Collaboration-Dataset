using UnityEngine;

public class ShowAnimatorFrame : MonoBehaviour
{
    public Animator animator;
    public AnimationClip clip;
    public int currentFrame;
    public float currentTime;
    public float normalizedTime;

    void Start()
    {
        if (animator == null)
            animator = GetComponent<Animator>();
    }

    void Update()
    {
        if (animator == null || clip == null)
            return;

        AnimatorStateInfo info = animator.GetCurrentAnimatorStateInfo(0);

        normalizedTime = info.normalizedTime % 1f;
        currentTime = normalizedTime * clip.length;
        currentFrame = Mathf.FloorToInt(currentTime * clip.frameRate);
    }
}*** Delete File: outputs/hand_refine_drill_shot005/HandPoseReplayer.cs
