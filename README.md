# MovieBench
### <div align="center"> A Hierarchical Movie Level Dataset for Long Video Generation <div> 

<div align="center">
  <a href="https://weijiawu.github.io/MovieBench/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://arxiv.org/abs/2411.15262"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv&color=red&logo=arxiv"></a> &ensp;
</div>

## :notes: **Updates**

- [x] Dec. 16, 2024. Release [DataSplit](https://github.com/showlab/MovieBench/blob/main/data/data_split.json), [Scene Split](https://github.com/showlab/MovieBench/blob/main/data/movies_scenes.json).
- [x] Dec. 16, 2024. Release the Scripts for Shot-Level Annotation Generation with GPT4.
- [x] Nov. 22, 2024. Rep initialization.


---
## :notes: **Todo**
- [ ] Release Dataset within the next three months.
- [ ] Building Leaderboard.
- [ ] Release Metric Scripts.

## 🐱 Abstract
<font color="red">MovieBench</font> is a Hierarchical Movie-Level Dataset for Long Video Generation, which addresses these challenges by providing unique contributions:
(1) movie-length videos featuring rich, coherent storylines and multi-scene narratives, (2) consistency of character appearance and audio across scenes, and (3) hierarchical data structure contains high-level movie information and detailed shot-level descriptions. Experiments demonstrate that MovieBench brings some new insights and challenges, such as maintaining character ID consistency across multiple scenes for various characters. The dataset will be public and continuously maintained, aiming
to advance the field of long video generation.

---

![image.](asset/structure.png)

---
![image.](asset/keyframeGen.png)


## ⏬ Download Data



## ⏬ Shot-Level Annotation Generation with GPT4

We developed our Shot-Level Annotation Generation system based on [MovieSeq](https://github.com/showlab/MovieSeq), leveraging GPT-4 to enhance its functionality.


<img src="asset/22.02.56.png" alt="image description" width="500" height="300" style="display: block; margin: 0 auto;">


Using a Visual Language Model (e.g., GPT-4), you can generate detailed annotations that include the following elements:
```
{
    "Characters":
    {
        "Character Name 1": "Description for appearance and behavior of Character 1, within 30 words",
        "Character Name 2": "Description for appearance and behavior of Character 2, within 30 words", 
    },
    "Style Elements":
    [
        "Element 1", "Element 2", "Element 3"
    ],
    "Plot":"A concise summary focusing on the main event or emotion, within 80 words",
    "Background Description":"A concise summary focusing on the main event or emotion, within 40 words",
    "Camera Motion":"A concise summary focusing camera motion, within 30 words."
}
```

For detailed ```environment setup``` and ```usage instructions```, please refer to the corresponding [README](https://github.com/showlab/MovieBench/tree/main/MovieBench).


## 📖BibTeX
    @misc{wu2024moviebenchhierarchicalmovielevel,
      title={MovieBench: A Hierarchical Movie Level Dataset for Long Video Generation}, 
      author={Weijia Wu and Mingyu Liu and Zeyu Zhu and Xi Xia and Haoen Feng and Wen Wang and Kevin Qinghong Lin and Chunhua Shen and Mike Zheng Shou},
      year={2024},
      eprint={2411.15262},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.15262}, 
      }
    
## 🤗Acknowledgements
- Thanks to [Diffusers](https://github.com/huggingface/diffusers) for the wonderful work.
