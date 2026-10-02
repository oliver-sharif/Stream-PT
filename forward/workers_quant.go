//go:build goexperiment.simd

package forward

import "sync"

type quantRowJob struct {
	data                 []byte
	firstRow, begin, end int
}

// quantWorkers owns workers for one multiplication, not for one mmap window.
type quantWorkers struct {
	jobs    []chan quantRowJob
	window  sync.WaitGroup
	joined  sync.WaitGroup
	compute func(int, quantRowJob)
}

func newQuantWorkers(count int, compute func(int, quantRowJob)) *quantWorkers {
	p := &quantWorkers{compute: compute}
	if count == 1 {
		return p
	}
	p.jobs = make([]chan quantRowJob, count)
	p.joined.Add(count)
	for worker := range p.jobs {
		p.jobs[worker] = make(chan quantRowJob)
		go func() {
			defer p.joined.Done()
			for job := range p.jobs[worker] {
				p.compute(worker, job)
				// Drop the mapped slice before acknowledging completion.
				job = quantRowJob{}
				p.window.Done()
			}
		}()
	}
	return p
}

func (p *quantWorkers) run(data []byte, firstRow, rows int) {
	if len(p.jobs) == 0 {
		p.compute(0, quantRowJob{data: data, firstRow: firstRow, end: rows})
		return
	}
	active := min(len(p.jobs), rows)
	p.window.Add(active)
	for worker := 0; worker < active; worker++ {
		p.jobs[worker] <- quantRowJob{data: data, firstRow: firstRow,
			begin: worker * rows / active, end: (worker + 1) * rows / active}
	}
	// Never return from the mmap callback on cancellation before this barrier.
	p.window.Wait()
}

func (p *quantWorkers) close() {
	for _, jobs := range p.jobs {
		close(jobs)
	}
	p.joined.Wait()
}
