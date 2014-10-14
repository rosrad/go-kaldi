package kaldi

import (
	"fmt"
	"path"
	"strconv"
)

type TrainConf struct {
	MinBatch int
	Jobs     int
	Layers   int
	Context  int
}

func NewTrainConf() *TrainConf {
	return &TrainConf{
		MinBatch: 512,
		Jobs:     4,
		Layers:   5,
		Context:  4}
}

type Bottleneck struct {
	Tag     string
	Target  string
	Align   string
	feature string
	Dynamic string
}

func NewBottleneck() *Bottleneck {
	return &Bottleneck{"test", "tri1", "tri1", "mfcc", "delta"}
}

func (b *Bottleneck) AlignDir() string {
	return path.Join(NewPathConf(b.feature).Exp, JoinParams(b.Align, "ali"))
}

func (b *Bottleneck) TargetName() string {
	return JoinParams(b.Target, b.Tag)
}

func (b *Bottleneck) TargetDir() string {
	return path.Join(BnfConf().FeatExp, b.TargetName())
}

func (b *Bottleneck) Train(conf *TrainConf) bool {
	cmd_str := JoinArgs(
		"steps/nnet2/train_tanh_bottleneck.sh",
		" --stage -100",
		"--num-jobs-nnet "+strconv.Itoa(conf.Jobs),
		"--num-threads 1 ",
		"--mix-up 5000",
		"--max-change 40",
		"--splice-width "+strconv.Itoa(conf.Context),
		"--minibatch-size "+strconv.Itoa(conf.MinBatch),
		"--initial-learning-rate 0.005",
		"--final-learning-rate 0.0005",
		"--num-hidden-layers 5",
		"--bottleneck-dim 42",
		"--hidden-layer-dim "+strconv.Itoa(conf.Layers),
		MfccTrain(),
		Lang(),
		b.AlignDir(),
		b.TargetDir())

	fmt.Println(cmd_str)
	BashRun(cmd_str)
	return true
}

func (b *Bottleneck) Conf() *PathConf {
	return BnfConf().MixTag(b.TargetName())
}

func (b *Bottleneck) CleanStorage() bool {

	return BnfConf().Bakup(b.Conf()) != 0
}

func (b *Bottleneck) InsureStorage() {
	b.CleanStorage()
	dirs := []string{b.Conf().Exp, b.Conf().FeatData, b.Conf().FeatParam}
	for _, dir := range dirs {
		InsureDir(dir)
	}

}

func (b *Bottleneck) SwitchTo() {
	cmd_str := JoinArgs(
		"bnf_switch.sh",
		b.TargetName())
	fmt.Println(cmd_str)
	BashRun(cmd_str)
}

func (b *Bottleneck) Dump(set string) bool {

	dirs, err := Subsets(set, "mfcc")
	if err != nil {
		fmt.Println(err)
		return false
	}

	for _, dir := range dirs {
		cmd_str := JoinArgs(
			"steps/nnet2/dump_bottleneck_features.sh",
			"--nj 10",
			path.Join(MfccConf().FeatData, dir),
			path.Join(b.Conf().FeatData, dir),
			b.TargetDir(),
			path.Join(b.Conf().FeatParam, dir),
			b.Conf().FeatDump)
		fmt.Println(cmd_str)
		BashRun(cmd_str)
	}
	return true
}

func (b *Bottleneck) DumpSets(sets []string) {
	b.InsureStorage()
	for _, set := range sets {
		b.Dump(set)
	}
}
